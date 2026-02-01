from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ----------------------------
# Parsing / settings helpers
# ----------------------------


def parse_portfolio_lines(raw: str) -> tuple[list[tuple[str, float]], str | None]:
    """
    Parse portfolio input: TICKER, WEIGHT (one per line).
    Returns: (list of (ticker, weight), error_message)
    """
    lines = [ln.strip() for ln in (raw or "").splitlines()]
    portfolio = []
    seen_tickers = set()

    for ln in lines:
        if not ln:
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 2:
            return [], f"Invalid format: '{ln}'. Expected: TICKER, WEIGHT"

        ticker = parts[0].upper()
        try:
            weight = float(parts[1])
        except ValueError:
            return [], f"Invalid weight for {ticker}: '{parts[1]}' is not a number"

        if weight < 0:
            return [], f"Weight for {ticker} must be non-negative (got {weight})"

        if ticker in seen_tickers:
            return [], f"Duplicate ticker: {ticker}"

        seen_tickers.add(ticker)
        portfolio.append((ticker, weight))

    if not portfolio:
        return [], "Portfolio is empty"

    total_weight = sum(w for _, w in portfolio)
    if abs(total_weight - 100.0) > 0.01:
        return [], f"Weights must sum to 100% (currently {total_weight:.2f}%)"

    return portfolio, None


def parse_benchmark_tickers(raw: str) -> list[str]:
    """Parse comma-separated benchmark tickers."""
    if not raw:
        return []
    tickers = [t.strip().upper() for t in raw.split(",")]
    return [t for t in tickers if t]


def resolve_date_preset(preset: str, start_custom: date, end_custom: date) -> tuple[date, date]:
    today = date.today()
    if preset == "Custom":
        return start_custom, end_custom
    if preset == "Last 3 months":
        return today - timedelta(days=90), today
    if preset == "Last 6 months":
        return today - timedelta(days=180), today
    if preset == "YTD":
        return date(today.year, 1, 1), today
    if preset == "Last 12 months":
        return today - timedelta(days=365), today
    if preset == "Last 24 months":
        return today - timedelta(days=730), today
    if preset == "Last 36 months":
        return today - timedelta(days=1095), today
    return start_custom, end_custom


def fmt_d(d: date) -> str:
    return d.strftime("%d/%m/%Y")


# ----------------------------
# Yahoo metadata + currency
# ----------------------------

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_symbol_meta(symbol: str) -> dict:
    """Best-effort metadata (cached): currency + name."""
    t = yf.Ticker(symbol)
    meta = {"symbol": symbol, "currency": None, "name": None}
    try:
        meta["currency"] = getattr(t.fastinfo, "currency", None)
    except Exception:
        meta["currency"] = None

    if not meta["currency"]:
        try:
            meta["currency"] = (t.info or {}).get("currency", None)
        except Exception:
            meta["currency"] = None

    try:
        info = t.info or {}
        meta["name"] = info.get("longName") or info.get("shortName")
    except Exception:
        meta["name"] = None

    return meta


def currency_factor_to_major_units(yahoo_currency: str | None) -> tuple[float, str]:
    """Convert Yahoo 'GBp'/'GBX' (pence) into pounds-equivalent major units."""
    if yahoo_currency in {"GBp", "GBX"}:
        return 1.0 / 100.0, "Converted from pence (GBp/GBX) to pounds-equivalent by /100"
    if yahoo_currency == "GBP":
        return 1.0, "Already in pounds (GBP)"
    if yahoo_currency:
        return 1.0, f"No conversion applied (Yahoo currency: {yahoo_currency})"
    return 1.0, "Currency unknown (no conversion applied)"


# ----------------------------
# Yahoo download (Close only)
# ----------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_yahoo_close(symbols: list[str], start: date, end: date) -> tuple[pd.DataFrame, list[dict]]:
    """Download auto-adjusted Close series for symbols."""
    if not symbols:
        return pd.DataFrame(), [{"symbol": "", "problem": "No symbols provided"}]

    end_plus = end + timedelta(days=1)
    data = yf.download(
        symbols,
        start=start,
        end=end_plus,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    issues: list[dict] = []

    if data is None or getattr(data, "empty", True):
        issues.append({"symbol": ",".join(symbols),
                      "problem": "No data returned for any symbol"})
        return pd.DataFrame(), issues

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            issues.append({"symbol": ",".join(
                symbols), "problem": "Expected Close in yfinance output but not found"})
            return pd.DataFrame(), issues
        close = data["Close"].copy()
    else:
        if "Close" in data.columns:
            close = data["Close"].to_frame()
            close.columns = [symbols[0]]
        elif "Adj Close" in data.columns:
            close = data["Adj Close"].to_frame()
            close.columns = [symbols[0]]
        else:
            issues.append(
                {"symbol": symbols[0], "problem": "Neither Close nor Adj Close found in yfinance output"})
            return pd.DataFrame(), issues

    for s in symbols:
        if s not in close.columns:
            close[s] = np.nan
            issues.append(
                {"symbol": s, "problem": "Symbol missing from Yahoo download (column added as all-NaN)"})

    close = close.sort_index()
    close = close[symbols]
    return close, issues


# ----------------------------
# Missing-history handling
# ----------------------------

def backfill_leading_flat(close: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Backfill leading NaN regions with first valid price."""
    close = close.sort_index().copy()
    missing_ranges: list[dict] = []

    if close.empty:
        return close, missing_ranges

    first_idx = close.index.min()
    last_idx = close.index.max()

    for s in close.columns:
        ser = close[s]
        if ser.isna().all():
            missing_ranges.append({
                "symbol": s,
                "type": "nodata",
                "start": first_idx,
                "end": last_idx,
            })
            continue

        first_valid = ser.first_valid_index()
        if first_valid is not None and first_valid > first_idx:
            missing_ranges.append({
                "symbol": s,
                "type": "leadingnan",
                "start": first_idx,
                "end": first_valid - pd.Timedelta(days=1),
                "used_price_from": first_valid,
            })
            fillvalue = close.at[first_valid, s]
            close.loc[(close.index >= first_idx) & (
                close.index < first_valid), s] = fillvalue

    close = close.ffill()
    return close, missing_ranges


# ----------------------------
# Spike cleaning
# ----------------------------

def clean_daily_spikes_flat(close: pd.DataFrame, threshold: float = 0.25) -> tuple[pd.DataFrame, list[dict]]:
    """Replace spikes exceeding threshold with previous day's price."""
    close = close.sort_index().copy()
    corrections: list[dict] = []

    if close.empty:
        return close, corrections

    for sym in close.columns:
        s = close[sym]
        prev_val = None

        for ts in s.index:
            val = s.at[ts]
            if pd.isna(val):
                continue

            if prev_val is None:
                prev_val = float(val)
                continue

            if prev_val == 0:
                prev_val = float(val)
                continue

            pct = (float(val) / prev_val) - 1.0
            if np.isfinite(pct) and abs(pct) > threshold:
                old = float(val)
                new = float(prev_val)
                s.at[ts] = new
                corrections.append({
                    "symbol": sym,
                    "date": ts,
                    "pct_move": pct,
                    "old_price": old,
                    "new_price": new,
                })
            else:
                prev_val = float(val)

        close[sym] = s

    return close, corrections


# ----------------------------
# Portfolio & Benchmark calculations
# ----------------------------

def calculate_portfolio_value(
    close: pd.DataFrame,
    portfolio: list[tuple[str, float]],
    initial_value: float = 1_000_000.0
) -> pd.Series:
    """Calculate portfolio value over time based on initial allocation."""
    if close.empty:
        return pd.Series(dtype=float)

    start_prices = close.iloc[0]
    holdings = {}

    for ticker, weight in portfolio:
        if ticker not in close.columns:
            continue
        if pd.isna(start_prices[ticker]) or start_prices[ticker] <= 0:
            continue
        allocation = initial_value * (weight / 100.0)
        holdings[ticker] = allocation / start_prices[ticker]

    portfolio_values = pd.Series(0.0, index=close.index)

    for ticker, num_holdings in holdings.items():
        portfolio_values += close[ticker] * num_holdings

    return portfolio_values


def calculate_benchmark_value(
    close: pd.DataFrame,
    ticker: str,
    initial_value: float = 1_000_000.0
) -> pd.Series:
    """Calculate benchmark value over time."""
    if ticker not in close.columns:
        return pd.Series(dtype=float)

    prices = close[ticker]
    start_price = prices.iloc[0]

    if pd.isna(start_price) or start_price <= 0:
        return pd.Series(dtype=float)

    num_holdings = initial_value / start_price
    return prices * num_holdings


def calculate_cash_value(
    date_range: pd.DatetimeIndex,
    rate_pct: float,
    initial_value: float = 1_000_000.0
) -> pd.Series:
    """Calculate cash value with daily compounding."""
    daily_rate = rate_pct / 100.0 / 365.0
    days_from_start = (date_range - date_range[0]).days
    values = initial_value * ((1 + daily_rate) ** days_from_start)
    return pd.Series(values, index=date_range)


def apply_inflation_adjustment(
    series: pd.Series,
    inflation_pct: float
) -> pd.Series:
    """Apply inflation adjustment using real return formula."""
    daily_inflation = inflation_pct / 100.0 / 365.0
    days_from_start = (series.index - series.index[0]).days
    inflation_factor = (1 + daily_inflation) ** days_from_start
    return series / inflation_factor


# ----------------------------
# Transformations for charting
# ----------------------------

def compute_rebased_index(values_df: pd.DataFrame, base_value: float = 100.0) -> pd.DataFrame:
    """Rebase all series to start at base_value."""
    out = {}
    for c in values_df.columns:
        s = values_df[c].dropna()
        if s.empty or s.iloc[0] == 0:
            continue
        out[c] = (s / s.iloc[0]) * base_value

    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_index()


def compute_cum_return(values_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative return for all series."""
    out = {}
    for c in values_df.columns:
        s = values_df[c].dropna()
        if s.empty or s.iloc[0] == 0:
            continue
        out[c] = (s / s.iloc[0]) - 1.0

    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_index()


# ----------------------------
# Plot + export
# ----------------------------

def plot_lines(df: pd.DataFrame, title: str, y_label: str, percent: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for col in df.columns:
        y = df[col] * 100.0 if percent else df[col]
        ax.plot(df.index, y, label=col, linewidth=1.6)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=2, fontsize=9)
    fig.autofmt_xdate()
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ----------------------------
# Notes generation
# ----------------------------

def build_notes_lines(
    start_date: date,
    end_date: date,
    missing_ranges: list[dict],
    corrections: list[dict],
    spike_threshold_pct: int,
    max_dates_per_symbol: int = 12,
) -> list[str]:
    """Generate human-readable notes about data quality."""
    lines: list[str] = []

    if not missing_ranges:
        lines.append(
            f"No prices missing for the period {fmt_d(start_date)} to {fmt_d(end_date)}.")
    else:
        any_leading = any(
            m.get("type") == "leadingnan" for m in missing_ranges)
        any_nodata = any(m.get("type") == "nodata" for m in missing_ranges)

        if any_leading:
            for m in missing_ranges:
                if m.get("type") != "leadingnan":
                    continue
                s = m["symbol"]
                start = pd.to_datetime(m["start"]).date()
                end = pd.to_datetime(m["end"]).date()
                used = pd.to_datetime(m["used_price_from"]).date()
                lines.append(
                    f"{s} missing prices from {fmt_d(start)} to {fmt_d(end)}; "
                    f"backfilled using the price from {fmt_d(used)} (assumed zero growth)."
                )

        if any_nodata:
            for m in missing_ranges:
                if m.get("type") != "nodata":
                    continue
                s = m["symbol"]
                start = pd.to_datetime(m["start"]).date()
                end = pd.to_datetime(m["end"]).date()
                lines.append(
                    f"{s} has no usable Yahoo price history between {fmt_d(start)} and {fmt_d(end)}.")

    if not corrections:
        lines.append(
            f"No spike flattening was applied (threshold {spike_threshold_pct}% daily move).")
    else:
        lines.append(
            f"Possible data-quality spikes were flattened when the 1-day move exceeded {spike_threshold_pct}% "
            "(today's price replaced with yesterday's)."
        )

        by_sym: dict[str, list[date]] = {}
        for c in corrections:
            sym = c["symbol"]
            d = pd.to_datetime(c["date"]).date()
            by_sym.setdefault(sym, []).append(d)

        for sym in sorted(by_sym.keys()):
            dts = sorted(set(by_sym[sym]))
            shown = dts[:max_dates_per_symbol]
            dates_str = ", ".join(dt.strftime("%d/%m/%Y") for dt in shown)
            more = "" if len(
                dts) <= max_dates_per_symbol else f" (+{len(dts) - max_dates_per_symbol} more)"
            lines.append(f"{sym} flattened on: {dates_str}{more}.")

    return lines


# ----------------------------
# Streamlit App
# ----------------------------

st.set_page_config(
    page_title="Historical Portfolio Performance", layout="wide")
st.title("Historical Portfolio Performance")

with st.sidebar:
    st.header("Portfolio Definition")
    raw_portfolio = st.text_area(
        "Tickers + Weight",
        value="VHYL.L, 50\nVUAG.L, 30\nXDEV.L, 20",
        height=160,
        help="Enter tickers with weights (one per line). Format: TICKER, WEIGHT. Weights must sum to 100%.",
    )

    st.header("Benchmark & Baselines")
    benchmark_input = st.text_input(
        "Benchmark tickers (comma-separated)",
        value="^GSPC",
        help="Enter one or more benchmark tickers separated by commas. Each starts with £1m."
    )

    cash_rate = st.number_input(
        "Cash interest rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=3.5,
        step=0.1,
        help="Annual interest rate for cash baseline, compounded daily"
    )

    st.header("Inflation Adjustment")
    apply_inflation = st.checkbox(
        "Adjust for inflation",
        value=False,
        help="Apply real return calculation: (1 + r_real) = (1 + r_nominal) / (1 + inflation)"
    )

    inflation_rate = st.number_input(
        "Inflation rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=4.0,
        step=0.1,
        disabled=not apply_inflation,
        help="Annual inflation rate applied daily"
    )

    st.header("Date Range")
    date_preset = st.selectbox(
        "Date range",
        ["Custom", "Last 3 months", "Last 6 months", "YTD",
            "Last 12 months", "Last 24 months", "Last 36 months"],
        index=4,
    )

    today = date.today()
    default_start = today - timedelta(days=365)
    start_custom = st.date_input("Start date", value=default_start)
    end_custom = st.date_input("End date", value=today)

    start_date, end_date = resolve_date_preset(
        date_preset, start_custom, end_custom)

    st.header("Chart Settings")
    chart_mode = st.selectbox(
        "Chart mode",
        ["Cumulative return (%)", "Rebased index (start=100)"],
        index=0,
    )

    st.subheader("Yahoo spike cleaning")
    enable_spike_clean = st.checkbox(
        "Flatten suspicious 1-day spikes",
        value=True,
        help="If abs(1-day move) exceeds the threshold, replace that day's price with the prior day's price.",
    )

    spike_threshold_pct = st.slider(
        "Spike threshold (%)",
        min_value=5,
        max_value=80,
        value=25,
        step=5,
    )

    show_currency_table = st.checkbox(
        "Show currency / pence-pound handling", value=False)

    run = st.button("Update chart", type="primary")

if not run:
    st.info("Enter portfolio composition on the left, adjust settings, then click 'Update chart'.")
    st.stop()

# Parse portfolio
portfolio, portfolio_error = parse_portfolio_lines(raw_portfolio)
if portfolio_error:
    st.error(f"Portfolio error: {portfolio_error}")
    st.stop()

# Parse benchmarks
benchmarks = parse_benchmark_tickers(benchmark_input)
if not benchmarks:
    st.error("Please enter at least one benchmark ticker.")
    st.stop()

# Date validation
if end_date <= start_date:
    st.error("End date must be after start date.")
    st.stop()

# Collect all symbols
portfolio_tickers = [t for t, _ in portfolio]
all_symbols = list(set(portfolio_tickers + benchmarks))

# Download prices
with st.spinner("Downloading prices from Yahoo..."):
    close_raw, issues = fetch_yahoo_close(all_symbols, start_date, end_date)

if close_raw.empty:
    st.error("No price data returned.")
    st.stop()

# Currency conversion
meta_rows = []
factors = {}
for s in all_symbols:
    meta = get_symbol_meta(s)
    factor, note = currency_factor_to_major_units(meta.get("currency"))
    factors[s] = factor
    meta_rows.append({
        "Symbol": s,
        "Name": meta.get("name") or "",
        "Yahoo currency": meta.get("currency") or "",
        "Applied factor": factor,
        "Note": note,
    })

meta_df = pd.DataFrame(meta_rows)

close = close_raw.copy()
for s, f in factors.items():
    if s in close.columns and f != 1.0:
        close[s] = close[s] * f

# Show download issues
if issues:
    st.warning("Some Yahoo download issues occurred:")
    for it in issues:
        st.write(
            f"- {it.get('symbol', '?')}: {it.get('problem', 'Unknown issue')}")

# Backfill and clean
close_filled, missing_ranges = backfill_leading_flat(close)
close_filled = close_filled.dropna(axis=1, how="all")

corrections: list[dict] = []
if enable_spike_clean:
    close_filled, corrections = clean_daily_spikes_flat(
        close_filled,
        threshold=float(spike_threshold_pct) / 100.0,
    )

# Verify all needed symbols are available
missing_portfolio = [
    t for t in portfolio_tickers if t not in close_filled.columns]
if missing_portfolio:
    st.error(f"Portfolio tickers unavailable: {', '.join(missing_portfolio)}")
    st.stop()

missing_benchmarks = [b for b in benchmarks if b not in close_filled.columns]
if missing_benchmarks:
    st.error(f"Benchmark tickers unavailable: {', '.join(missing_benchmarks)}")
    st.stop()

if show_currency_table:
    with st.expander("Currency + pence/pounds handling", expanded=False):
        st.dataframe(meta_df, use_container_width=True)

# Calculate all series
values_df = pd.DataFrame(index=close_filled.index)

# Portfolio
portfolio_values = calculate_portfolio_value(
    close_filled, portfolio, initial_value=1_000_000.0)
if not portfolio_values.empty:
    values_df["Portfolio"] = portfolio_values

# Benchmarks
for bench in benchmarks:
    bench_values = calculate_benchmark_value(
        close_filled, bench, initial_value=1_000_000.0)
    if not bench_values.empty:
        values_df[f"Benchmark: {bench}"] = bench_values

# Cash
cash_values = calculate_cash_value(
    close_filled.index, cash_rate, initial_value=1_000_000.0)
values_df[f"Cash ({cash_rate}%)"] = cash_values

# Apply inflation if selected
if apply_inflation:
    for col in values_df.columns:
        values_df[col] = apply_inflation_adjustment(
            values_df[col], inflation_rate)

# Transform for charting
if chart_mode == "Cumulative return (%)":
    plot_df = compute_cum_return(values_df)
    title = f"Cumulative return — {start_date} to {end_date}"
    if apply_inflation:
        title += f" (Inflation-adjusted @ {inflation_rate}%)"
    ylab = "Cumulative return (%)"
    percent = True
else:
    plot_df = compute_rebased_index(values_df, base_value=100.0)
    title = f"Rebased index (start=100) — {start_date} to {end_date}"
    if apply_inflation:
        title += f" (Inflation-adjusted @ {inflation_rate}%)"
    ylab = "Index level"
    percent = False

if plot_df.empty:
    st.error("Unable to generate chart data.")
    st.stop()

# Plot
fig = plot_lines(plot_df, title=title, y_label=ylab, percent=percent)
st.pyplot(fig, use_container_width=True)

png_bytes = fig_to_png_bytes(fig)
st.download_button(
    "Download chart as PNG",
    data=png_bytes,
    file_name=f"portfolio_chart_{start_date}_{end_date}.png",
    mime="image/png",
)

# Notes
st.subheader("Notes / data quality")
notes = build_notes_lines(
    start_date=start_date,
    end_date=end_date,
    missing_ranges=missing_ranges,
    corrections=corrections,
    spike_threshold_pct=spike_threshold_pct,
    max_dates_per_symbol=12,
)

for line in notes:
    st.write(f"- {line}")
