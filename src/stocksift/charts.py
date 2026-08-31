"""Plotly price charts for stocksift close-price data.

Input format:
    date | 005930 | 000660 | ...

Ichimoku is approximate because the source data contains closes only.
Weekly pseudo-high/low values use the max/min daily closes in each week.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


__all__ = [
    "plot_price_chart",
    "browse_price_chart",
    "plot_price_comparison",
    "browse_price_comparison",
]

_INDICATOR_ORDER = ("ma", "bb", "rsi", "ichimoku")
_INDICATORS = set(_INDICATOR_ORDER)

_CONTROL_GROUP_SPACING = 24
_INDICATOR_SPACING = 6


def plot_price_chart(
    prices: pd.DataFrame,
    ticker: str,
    *,
    ticker_names: Mapping[str, str] | None = None,
    date_col: str = "date",
    freq: str = "D",
    indicators: Iterable[str] = ("ma", "bb", "rsi", "ichimoku"),
    visible_indicators: Iterable[str] = ("ma", "bb", "rsi"),
    ma_windows: Sequence[int] = (20, 60),
    bb_window: int = 20,
    bb_std: float = 2.0,
    rsi_window: int = 14,
    ichimoku_windows: tuple[int, int, int] = (9, 26, 52),
    ichimoku_displacement: int = 26,
    price_line_width: float = 2.0,
    indicator_line_width: float = 1.0,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Return a technical chart for one ticker.

    RSI is visible by default. MA, Bollinger, RSI, and approximate Ichimoku
    share ``indicator_line_width``; the close price uses ``price_line_width``.

    The Plotly range selector only changes the visible x-range.
    It does not recalculate technical indicators.
    """
    freq = _check_freq(freq)
    indicators = _check_indicators(indicators)
    visible = _check_indicators(visible_indicators)

    if not visible <= indicators:
        raise ValueError("visible_indicators must be included in indicators")

    _check_windows(ma_windows, "ma_windows")
    _check_windows((bb_window, rsi_window), "indicator windows")
    _check_windows(ichimoku_windows, "ichimoku_windows")
    _check_windows((ichimoku_displacement,), "ichimoku_displacement")
    _check_positive_number(price_line_width, "price_line_width")
    _check_positive_number(indicator_line_width, "indicator_line_width")
    _check_figure_size(width, "width")
    _check_figure_size(height, "height")

    if len(ichimoku_windows) != 3:
        raise ValueError("ichimoku_windows must contain exactly 3 values")
    if not isinstance(bb_std, (int, float)) or isinstance(bb_std, bool) or bb_std <= 0:
        raise ValueError("bb_std must be positive")

    close = _price_series(prices, ticker, date_col=date_col)
    data = _price_periods(close, freq=freq)
    has_rsi = "rsi" in indicators

    fig = (
        make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.78, 0.22],
        )
        if has_rsi
        else make_subplots(rows=1, cols=1)
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["close"],
            mode="lines",
            name="Close",
            line=dict(width=price_line_width),
        ),
        row=1,
        col=1,
    )

    if "ma" in indicators:
        state = _visibility("ma", visible)
        for window in ma_windows:
            values = data["close"].rolling(window).mean()
            fig.add_trace(
                go.Scatter(
                    x=values.index,
                    y=values,
                    mode="lines",
                    name=f"MA{window}",
                    line=dict(width=indicator_line_width),
                    visible=state,
                ),
                row=1,
                col=1,
            )

    if "bb" in indicators:
        state = _visibility("bb", visible)
        bb = _bollinger(
            data["close"],
            window=bb_window,
            n_std=float(bb_std),
        )

        fig.add_trace(
            go.Scatter(
                x=bb.index,
                y=bb["upper"],
                mode="lines",
                name=f"Bollinger ({bb_window}, {bb_std:g})",
                line=dict(width=indicator_line_width),
                legendgroup="bb",
                visible=state,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bb.index,
                y=bb["lower"],
                mode="lines",
                name="Bollinger lower",
                line=dict(width=indicator_line_width),
                legendgroup="bb",
                showlegend=False,
                fill="tonexty",
                visible=state,
            ),
            row=1,
            col=1,
        )

    if "ichimoku" in indicators:
        state = _visibility("ichimoku", visible)
        ichi = _ichimoku(
            data,
            windows=ichimoku_windows,
            displacement=ichimoku_displacement,
        )

        traces = [
            ("tenkan", "Approx. Ichimoku", True, None),
            ("kijun", "Kijun", False, None),
            ("senkou_a", "Senkou A", False, None),
            ("senkou_b", "Senkou B", False, "tonexty"),
            ("chikou", "Chikou", False, None),
        ]
        for column, name, showlegend, fill in traces:
            fig.add_trace(
                go.Scatter(
                    x=ichi.index,
                    y=ichi[column],
                    mode="lines",
                    name=name,
                    line=dict(width=indicator_line_width),
                    legendgroup="ichimoku",
                    showlegend=showlegend,
                    fill=fill,
                    visible=state,
                ),
                row=1,
                col=1,
            )

    if has_rsi:
        state = _visibility("rsi", visible)
        rsi = _rsi(data["close"], window=rsi_window)

        fig.add_trace(
            go.Scatter(
                x=rsi.index,
                y=rsi,
                mode="lines",
                name=f"RSI{rsi_window}",
                line=dict(width=indicator_line_width),
                legendgroup="rsi",
                visible=state,
            ),
            row=2,
            col=1,
        )

        for level in (70, 30):
            fig.add_trace(
                go.Scatter(
                    x=rsi.index,
                    y=[level] * len(rsi),
                    mode="lines",
                    line=dict(
                        width=indicator_line_width,
                        dash="dot",
                    ),
                    name=f"RSI {level}",
                    legendgroup="rsi",
                    showlegend=False,
                    hoverinfo="skip",
                    visible=state,
                ),
                row=2,
                col=1,
            )

        fig.update_yaxes(
            title_text="RSI",
            range=[0, 100],
            row=2,
            col=1,
        )

    label = _ticker_label(ticker, ticker_names)
    freq_label = "Daily" if freq == "D" else "Weekly"

    if height is None:
        height = 760 if has_rsi else 600

    fig.update_layout(
        title=f"{label} — {freq_label}",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            itemclick=False,
            itemdoubleclick=False,
        ),
        width=width,
        height=height,
        margin=dict(l=50, r=30, t=60, b=40),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)

    _add_time_controls(
        fig,
        row=2 if has_rsi else 1,
        col=1,
    )

    return fig


def browse_price_chart(
    prices: pd.DataFrame,
    tickers: Iterable[str],
    *,
    ticker_names: Mapping[str, str] | None = None,
    date_col: str = "date",
    freq: str = "D",
    **plot_kwargs: Any,
) -> None:
    """Browse technical charts with persistent indicator checkboxes.

    Ticker, frequency, and indicator controls are treated as three UI groups.
    The spacing between those groups is equal. Indicator checkbox state is
    preserved when the ticker or frequency changes.
    """
    widgets, display, clear_output = _notebook_tools()
    tickers = _check_tickers(prices, tickers, date_col=date_col)
    freq = _check_freq(freq)

    available_indicators = _check_indicators(
        plot_kwargs.pop("indicators", _INDICATOR_ORDER)
    )
    initial_visible = _check_indicators(
        plot_kwargs.pop(
            "visible_indicators",
            ("ma", "bb", "rsi"),
        )
    )

    if not initial_visible <= available_indicators:
        raise ValueError(
            "visible_indicators must be included in indicators"
        )

    ticker_selector = widgets.Dropdown(
        options=[
            (_ticker_label(ticker, ticker_names), ticker)
            for ticker in tickers
        ],
        value=tickers[0],
        description="",
        layout=widgets.Layout(width="300px"),
    )

    freq_selector = widgets.ToggleButtons(
        options=[
            ("Daily", "D"),
            ("Weekly", "W"),
        ],
        value=freq,
        description="",
    )

    indicator_labels = {
        "ma": "MA",
        "bb": "BB",
        "rsi": "RSI",
        "ichimoku": "Ichimoku",
    }
    indicator_checks = {}

    for indicator in _INDICATOR_ORDER:
        if indicator not in available_indicators:
            continue

        indicator_checks[indicator] = widgets.Checkbox(
            value=indicator in initial_visible,
            description=indicator_labels[indicator],
            indent=False,
            layout=widgets.Layout(
                width="auto",
                margin=f"0 {_INDICATOR_SPACING}px 0 0",
            ),
        )

    indicator_controls = widgets.HBox(
        list(indicator_checks.values()),
        layout=widgets.Layout(
            align_items="center",
        ),
    )

    group_margin = f"0 {_CONTROL_GROUP_SPACING}px 0 0"

    ticker_group = widgets.Box(
        [ticker_selector],
        layout=widgets.Layout(
            margin=group_margin,
        ),
    )
    freq_group = widgets.Box(
        [freq_selector],
        layout=widgets.Layout(
            margin=group_margin,
        ),
    )
    indicator_group = widgets.Box(
        [indicator_controls],
    )

    controls = widgets.HBox(
        [
            ticker_group,
            freq_group,
            indicator_group,
        ],
        layout=widgets.Layout(
            align_items="center",
            flex_flow="row",
        ),
    )
    output = widgets.Output()

    def render(*_: object) -> None:
        selected_indicators = tuple(
            indicator
            for indicator in _INDICATOR_ORDER
            if (
                indicator in indicator_checks
                and indicator_checks[indicator].value
            )
        )

        with output:
            clear_output(wait=True)
            plot_price_chart(
                prices,
                ticker_selector.value,
                ticker_names=ticker_names,
                date_col=date_col,
                freq=freq_selector.value,
                indicators=selected_indicators,
                visible_indicators=selected_indicators,
                **plot_kwargs,
            ).show()

    ticker_selector.observe(render, names="value")
    freq_selector.observe(render, names="value")

    for checkbox in indicator_checks.values():
        checkbox.observe(render, names="value")

    render()

    display(widgets.VBox([controls, output]))

def plot_price_comparison(
    prices: pd.DataFrame,
    tickers: Iterable[str],
    *,
    ticker_names: Mapping[str, str] | None = None,
    date_col: str = "date",
    freq: str = "D",
    base: float | None = 100.0,
    price_line_width: float = 2.0,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Compare tickers using raw or normalized closes from a common start date."""
    freq = _check_freq(freq)
    tickers = _check_tickers(prices, tickers, date_col=date_col)
    _check_positive_number(price_line_width, "price_line_width")
    _check_figure_size(width, "width")
    _check_figure_size(height, "height")

    if base is not None:
        _check_positive_number(base, "base")

    data = prices[[date_col, *tickers]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    if data[date_col].isna().any():
        raise ValueError(f"{date_col!r} contains invalid dates")

    data = (
        data.drop_duplicates(date_col, keep="last")
        .sort_values(date_col)
        .set_index(date_col)
    )
    data = data.apply(pd.to_numeric, errors="coerce")

    if freq == "W":
        data = data.resample("W-FRI").last()

    data = data.dropna(subset=tickers)
    if data.empty:
        raise ValueError("no common valid dates for requested tickers")

    first = data.iloc[0]
    if (first <= 0).any():
        bad = first.index[first <= 0].tolist()
        raise ValueError(f"non-positive start prices for tickers: {bad}")

    if base is None:
        title_prefix = ""
        yaxis_title = "Price"
    else:
        data = data.div(first).mul(float(base))
        title_prefix = "Normalized "
        yaxis_title = f"Normalized Price (base={base:g})"

    fig = go.Figure()

    for ticker in tickers:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[ticker],
                mode="lines",
                name=_ticker_label(ticker, ticker_names),
                line=dict(width=price_line_width),
            )
        )

    freq_label = "Daily" if freq == "D" else "Weekly"

    if height is None:
        height = 600

    fig.update_layout(
        title=f"{title_prefix}Price Comparison — {freq_label}",
        hovermode="x unified",
        yaxis_title=yaxis_title,
        width=width,
        height=height,
        margin=dict(l=50, r=30, t=60, b=40),
    )

    _add_time_controls(fig)

    return fig


def browse_price_comparison(
    prices: pd.DataFrame,
    tickers: Iterable[str],
    *,
    ticker_names: Mapping[str, str] | None = None,
    date_col: str = "date",
    normalize: bool = True,
    base: float = 100.0,
    price_line_width: float = 2.0,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Browse multi-stock comparisons with top-aligned one-row controls.

    The ticker selector is fixed to four visible rows and scrolls internally
    when more tickers are available. Selected tickers are also visible in the
    Plotly legend.

    ``normalize`` controls the initial state of the Normalized/Raw UI toggle.
    Comparison browsing uses the default daily frequency; callers that need
    another frequency can call ``plot_price_comparison`` directly.
    When normalized, ``base`` is used as the common starting value.
    """
    widgets, display, clear_output = _notebook_tools()
    tickers = _check_tickers(prices, tickers, date_col=date_col)

    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a bool")

    _check_positive_number(base, "base")
    _check_positive_number(price_line_width, "price_line_width")
    _check_figure_size(width, "width")
    _check_figure_size(height, "height")

    ticker_selector = widgets.SelectMultiple(
        options=[
            (_ticker_label(ticker, ticker_names), ticker)
            for ticker in tickers
        ],
        value=tuple(tickers[: min(2, len(tickers))]),
        description="",
        rows=4,
        layout=widgets.Layout(width="300px"),
    )
    normalize_selector = widgets.ToggleButtons(
        options=[
            ("Normalized", True),
            ("Raw", False),
        ],
        value=normalize,
        description="",
    )
    group_margin = f"0 {_CONTROL_GROUP_SPACING}px 0 0"
    
    ticker_group = widgets.Box(
        [ticker_selector],
        layout=widgets.Layout(
            margin=group_margin,
        ),
    )
    
    normalize_group = widgets.Box(
        [normalize_selector],
    )
    
    controls = widgets.HBox(
        [
            ticker_group,
            normalize_group,
        ],
        layout=widgets.Layout(
            align_items="center",
            flex_flow="row",
        ),
    )
    output = widgets.Output()

    def render(*_: object) -> None:
        with output:
            clear_output(wait=True)
            selected = list(ticker_selector.value)

            if not selected:
                print("Select at least one ticker.")
                return

            plot_price_comparison(
                prices,
                selected,
                ticker_names=ticker_names,
                date_col=date_col,
                base=base if normalize_selector.value else None,
                price_line_width=price_line_width,
                width=width,
                height=height,
            ).show()

    ticker_selector.observe(render, names="value")
    normalize_selector.observe(render, names="value")
    render()

    display(widgets.VBox([controls, output]))


def _price_series(
    prices: pd.DataFrame,
    ticker: str,
    *,
    date_col: str,
) -> pd.Series:
    _check_tickers(prices, [ticker], date_col=date_col)

    data = prices[[date_col, ticker]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    if data[date_col].isna().any():
        raise ValueError(f"{date_col!r} contains invalid dates")

    data[ticker] = pd.to_numeric(data[ticker], errors="coerce")
    close = (
        data.drop_duplicates(date_col, keep="last")
        .sort_values(date_col)
        .set_index(date_col)[ticker]
        .dropna()
    )

    if close.empty:
        raise ValueError(f"{ticker!r} has no valid prices")
    if (close <= 0).any():
        raise ValueError(f"{ticker!r} contains non-positive prices")

    return close.rename("close")


def _price_periods(close: pd.Series, *, freq: str) -> pd.DataFrame:
    if freq == "D":
        return pd.DataFrame(
            {"high": close, "low": close, "close": close}
        )

    return close.resample("W-FRI").agg(
        high="max",
        low="min",
        close="last",
    ).dropna(subset=["close"])


def _bollinger(
    close: pd.Series,
    *,
    window: int,
    n_std: float,
) -> pd.DataFrame:
    middle = close.rolling(window).mean()
    std = close.rolling(window).std()
    return pd.DataFrame(
        {
            "upper": middle + n_std * std,
            "lower": middle - n_std * std,
        }
    )


def _rsi(close: pd.Series, *, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window,
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window,
    ).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    return rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)


def _ichimoku(
    data: pd.DataFrame,
    *,
    windows: tuple[int, int, int],
    displacement: int,
) -> pd.DataFrame:
    tenkan_n, kijun_n, senkou_b_n = windows
    high, low, close = data["high"], data["low"], data["close"]

    tenkan = (
        high.rolling(tenkan_n).max()
        + low.rolling(tenkan_n).min()
    ) / 2
    kijun = (
        high.rolling(kijun_n).max()
        + low.rolling(kijun_n).min()
    ) / 2

    return pd.DataFrame(
        {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": ((tenkan + kijun) / 2).shift(displacement),
            "senkou_b": (
                (
                    high.rolling(senkou_b_n).max()
                    + low.rolling(senkou_b_n).min()
                )
                / 2
            ).shift(displacement),
            "chikou": close.shift(-displacement),
        }
    )


def _add_time_controls(
    fig: go.Figure,
    *,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add range buttons inside the upper-left of the main plot area."""
    kwargs: dict[str, Any] = {
        "rangeslider_visible": False,
        "rangeselector": dict(
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            buttons=[
                dict(
                    count=3,
                    label="3M",
                    step="month",
                    stepmode="backward",
                ),
                dict(
                    count=6,
                    label="6M",
                    step="month",
                    stepmode="backward",
                ),
                dict(
                    count=1,
                    label="1Y",
                    step="year",
                    stepmode="backward",
                ),
                dict(
                    count=3,
                    label="3Y",
                    step="year",
                    stepmode="backward",
                ),
                dict(
                    label="ALL",
                    step="all",
                ),
            ],
        ),
    }

    if row is None or col is None:
        fig.update_xaxes(**kwargs)
    else:
        fig.update_xaxes(
            **kwargs,
            row=row,
            col=col,
        )

def _check_freq(freq: str) -> str:
    freq = str(freq).upper()
    if freq not in {"D", "W"}:
        raise ValueError("freq must be 'D' or 'W'")
    return freq


def _check_indicators(indicators: Iterable[str]) -> set[str]:
    if isinstance(indicators, str):
        result = {indicators.lower()}
    else:
        result = {str(item).lower() for item in indicators}

    unknown = result - _INDICATORS
    if unknown:
        raise ValueError(f"unsupported indicators: {sorted(unknown)}")

    return result


def _check_windows(values: Iterable[int], name: str) -> None:
    for value in values:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{name} must contain positive integers")


def _check_positive_number(value: object, name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive")


def _check_figure_size(value: int | None, name: str) -> None:
    if value is None:
        return

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer or None")


def _check_tickers(
    prices: pd.DataFrame,
    tickers: Iterable[str],
    *,
    date_col: str,
) -> list[str]:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if date_col not in prices.columns:
        raise ValueError(f"prices must contain {date_col!r}")

    if isinstance(tickers, str):
        result = [tickers]
    else:
        result = list(dict.fromkeys(tickers))

    if not result:
        raise ValueError("at least one ticker is required")
    if not all(isinstance(t, str) and t for t in result):
        raise ValueError("tickers must be non-empty strings")

    missing = [
        ticker
        for ticker in result
        if ticker not in prices.columns
    ]
    if missing:
        raise ValueError(f"tickers missing from prices: {missing}")

    return result


def _ticker_label(
    ticker: str,
    ticker_names: Mapping[str, str] | None,
) -> str:
    if ticker_names is None:
        return ticker

    name = ticker_names.get(ticker)
    if name is None or pd.isna(name) or not str(name).strip():
        return ticker

    return f"{ticker} | {name}"


def _visibility(
    indicator: str,
    visible_indicators: set[str],
) -> bool | str:
    return True if indicator in visible_indicators else "legendonly"


def _notebook_tools():
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError as exc:
        raise ImportError(
            "browse_* functions require ipywidgets and IPython"
        ) from exc

    return widgets, display, clear_output
