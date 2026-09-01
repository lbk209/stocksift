"""Plotly price charts for stocksift price data.

Supported input formats
-----------------------
Close-only wide:
    date | 005930 | 000660 | ...

OHLCV MultiIndex:
    index: (date, ticker)
    columns: open, high, low, close, volume

Close-only data uses pseudo high/low values for Ichimoku. OHLCV data uses
the actual high/low values. Weekly OHLCV aggregation uses first/max/min/last
for OHLC and sums volume.
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

<<<<<<< HEAD
_INDICATOR_ORDER = ("ma", "bb", "rsi", "ichimoku", "volume")
=======
_INDICATOR_ORDER = ("ma", "bb", "ichimoku", "volume", "rsi", "mfi", "atr")
>>>>>>> d1aad7a (candlestick)
_INDICATORS = set(_INDICATOR_ORDER)

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

_CONTROL_GROUP_SPACING = 12
_INDICATOR_SPACING = 6
_CONTROL_HEIGHT = "30px"

_PERIOD_OPTIONS = ("3M", "6M", "1Y", "3Y", "ALL")


def plot_price_chart(
    prices: pd.DataFrame,
    ticker: str,
    *,
    ticker_names: Mapping[str, str] | None = None,
    date_col: str = "date",
    freq: str = "D",
    period: str = "ALL",
<<<<<<< HEAD
    indicators: Iterable[str] = ("ma", "bb", "rsi", "ichimoku"),
    visible_indicators: Iterable[str] = ("ma", "bb", "rsi"),
=======
    indicators: Iterable[str] | None = None,
    visible_indicators: Iterable[str] | None = None,
>>>>>>> d1aad7a (candlestick)
    ma_windows: Sequence[int] = (20, 60),
    bb_window: int = 20,
    bb_std: float = 2.0,
    rsi_window: int = 14,
<<<<<<< HEAD
=======
    atr_window: int = 14,
    mfi_window: int = 14,
>>>>>>> d1aad7a (candlestick)
    ichimoku_windows: tuple[int, int, int] = (9, 26, 52),
    ichimoku_displacement: int = 26,
    price_line_width: float = 2.0,
    indicator_line_width: float = 1.0,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Return a technical chart for one ticker.

<<<<<<< HEAD
    RSI is visible by default. MA, Bollinger, RSI, and Ichimoku share
    ``indicator_line_width``; the close price uses ``price_line_width``.

    Technical indicators are calculated from the full available history,
    then the selected period is displayed. Close-only input uses approximate
    Ichimoku; OHLCV input uses actual high/low values and can display volume.
    """
    freq = _check_freq(freq)
    period = _check_period(period)
    indicators = _check_indicators(indicators)
    visible = _check_indicators(visible_indicators)
    supported = _available_indicators(prices, date_col=date_col)

    unsupported = indicators - supported
    if unsupported:
        raise ValueError(
            f"indicators unavailable for this price data: {sorted(unsupported)}"
        )
=======
    OHLCV input uses a candlestick chart; close-only input uses a close line.
    MA and Volume are visible by default for OHLCV data, while close-only data
    defaults to MA only. Bollinger, RSI, Ichimoku, ATR, and MFI are optional.

    Technical indicators are calculated from the full available history,
    then the selected period is displayed. Close-only input uses approximate
    Ichimoku; OHLCV input uses actual high/low values.
    """
    freq = _check_freq(freq)
    period = _check_period(period)
    supported = _available_indicators(prices, date_col=date_col)

    if indicators is None:
        indicators = {"ma", "volume"} & supported
    else:
        indicators = _check_indicators(indicators)
        unsupported = indicators - supported
        if unsupported:
            raise ValueError(
                "indicators unavailable for this price data: "
                f"{sorted(unsupported)}"
            )

    if visible_indicators is None:
        visible = set(indicators)
    else:
        visible = _check_indicators(visible_indicators)

>>>>>>> d1aad7a (candlestick)
    if not visible <= indicators:
        raise ValueError("visible_indicators must be included in indicators")

    _check_windows(ma_windows, "ma_windows")
<<<<<<< HEAD
    _check_windows((bb_window, rsi_window), "indicator windows")
=======
    _check_windows(
        (bb_window, rsi_window, atr_window, mfi_window),
        "indicator windows",
    )
>>>>>>> d1aad7a (candlestick)
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

    price_kind = _price_kind(prices, date_col=date_col)
    data = _ticker_price_periods(
        prices,
        ticker,
        date_col=date_col,
        freq=freq,
    )
    view = _slice_period(data, period=period)

    has_rsi = "rsi" in indicators
<<<<<<< HEAD
    has_volume = "volume" in indicators
    extra_rows = int(has_rsi) + int(has_volume)

    if extra_rows == 0:
        fig = make_subplots(rows=1, cols=1)
    elif extra_rows == 1:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.78, 0.22],
        )
    else:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.68, 0.17, 0.15],
        )

    rsi_row = 2 if has_rsi else None
    volume_row = (
        3 if has_rsi and has_volume
        else 2 if has_volume
        else None
    )

    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["close"],
            mode="lines",
            name="Close",
            line=dict(width=price_line_width),
        ),
        row=1,
        col=1,
    )

=======
    has_mfi = "mfi" in indicators
    has_atr = "atr" in indicators
    has_volume = "volume" in indicators

    extra_panels = (
        int(has_rsi)
        + int(has_mfi)
        + int(has_atr)
        + int(has_volume)
    )
    rows = 1 + extra_panels

    if rows == 1:
        fig = make_subplots(rows=1, cols=1)
    else:
        panel_height = 0.14
        main_height = 1.0 - panel_height * extra_panels
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.025,
            row_heights=[
                main_height,
                *([panel_height] * extra_panels),
            ],
        )

    next_row = 2
    rsi_row = None
    mfi_row = None
    atr_row = None
    volume_row = None

    if has_rsi:
        rsi_row = next_row
        next_row += 1
    if has_mfi:
        mfi_row = next_row
        next_row += 1
    if has_atr:
        atr_row = next_row
        next_row += 1
    if has_volume:
        volume_row = next_row

    if price_kind == "ohlcv":
        fig.add_trace(
            go.Candlestick(
                x=view.index,
                open=view["open"],
                high=view["high"],
                low=view["low"],
                close=view["close"],
                name="OHLC",
                increasing=dict(
                    line=dict(color="red"),
                    fillcolor="rgba(255, 0, 0, 0.35)",
                ),
                decreasing=dict(
                    line=dict(color="blue"),
                    fillcolor="rgba(0, 0, 255, 0.35)",
                ),
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=view.index,
                y=view["close"],
                mode="lines",
                name="Close",
                line=dict(width=price_line_width),
            ),
            row=1,
            col=1,
        )

>>>>>>> d1aad7a (candlestick)
    if "ma" in indicators:
        state = _visibility("ma", visible)
        for window in ma_windows:
            values = data["close"].rolling(window).mean()
            values = values.loc[view.index]
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
        bb = bb.loc[view.index]

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
        ichi = ichi.loc[view.index]

        ichimoku_name = (
            "Ichimoku" if price_kind == "ohlcv"
            else "Approx. Ichimoku"
        )
        traces = [
            ("tenkan", ichimoku_name, True, None),
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

<<<<<<< HEAD
=======
    if has_volume:
        state = _visibility("volume", visible)
        volume = view["volume"]

        fig.add_trace(
            go.Bar(
                x=volume.index,
                y=volume,
                name="Volume",
                visible=state,
            ),
            row=volume_row,
            col=1,
        )
        fig.update_yaxes(
            title_text="Volume",
            row=volume_row,
            col=1,
        )

>>>>>>> d1aad7a (candlestick)
    if has_rsi:
        state = _visibility("rsi", visible)
        rsi = _rsi(data["close"], window=rsi_window)
        rsi = rsi.loc[view.index]

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
<<<<<<< HEAD
            row=2,
=======
            row=rsi_row,
>>>>>>> d1aad7a (candlestick)
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
                row=rsi_row,
                col=1,
            )

        fig.update_yaxes(
            title_text="RSI",
            range=[0, 100],
            row=rsi_row,
            col=1,
        )

<<<<<<< HEAD
    if has_volume:
        state = _visibility("volume", visible)
        volume = view["volume"]

        fig.add_trace(
            go.Bar(
                x=volume.index,
                y=volume,
                name="Volume",
                visible=state,
            ),
            row=volume_row,
            col=1,
        )
        fig.update_yaxes(
            title_text="Volume",
            row=volume_row,
            col=1,
        )

=======
    if has_mfi:
        state = _visibility("mfi", visible)
        mfi = _mfi(data, window=mfi_window)
        mfi = mfi.loc[view.index]

        fig.add_trace(
            go.Scatter(
                x=mfi.index,
                y=mfi,
                mode="lines",
                name=f"MFI{mfi_window}",
                line=dict(width=indicator_line_width),
                legendgroup="mfi",
                visible=state,
            ),
            row=mfi_row,
            col=1,
        )

        for level in (80, 20):
            fig.add_trace(
                go.Scatter(
                    x=mfi.index,
                    y=[level] * len(mfi),
                    mode="lines",
                    line=dict(
                        width=indicator_line_width,
                        dash="dot",
                    ),
                    name=f"MFI {level}",
                    legendgroup="mfi",
                    showlegend=False,
                    hoverinfo="skip",
                    visible=state,
                ),
                row=mfi_row,
                col=1,
            )

        fig.update_yaxes(
            title_text="MFI",
            range=[0, 100],
            row=mfi_row,
            col=1,
        )

    if has_atr:
        state = _visibility("atr", visible)
        atr = _atr(data, window=atr_window)
        atr = atr.loc[view.index]

        fig.add_trace(
            go.Scatter(
                x=atr.index,
                y=atr,
                mode="lines",
                name=f"ATR{atr_window}",
                line=dict(width=indicator_line_width),
                legendgroup="atr",
                visible=state,
            ),
            row=atr_row,
            col=1,
        )
        
        fig.update_yaxes(
            title_text="ATR",
            row=atr_row,
            col=1,
        )


>>>>>>> d1aad7a (candlestick)
    label = _ticker_label(ticker, ticker_names)
    freq_label = "Daily" if freq == "D" else "Weekly"

    if height is None:
<<<<<<< HEAD
        if has_rsi and has_volume:
            height = 840
        elif has_rsi or has_volume:
            height = 760
        else:
            height = 600
=======
        height = 600 + 110 * extra_panels
>>>>>>> d1aad7a (candlestick)

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
<<<<<<< HEAD

=======
    fig.update_xaxes(rangeslider_visible=False)
>>>>>>> d1aad7a (candlestick)

    return fig


def browse_price_chart(
    prices: pd.DataFrame,
    tickers: Iterable[str],
    *,
    ticker_names: Mapping[str, str] | None = None,
    date_col: str = "date",
    freq: str = "D",
    period: str = "ALL",
    **plot_kwargs: Any,
) -> None:
    """Browse technical charts with persistent indicator checkboxes.

    Ticker, frequency, indicator, and period controls are separate UI groups.
    The spacing between groups is equal. Indicator checkbox state is preserved
    when the ticker, frequency, or period changes.
    """
    widgets, display, clear_output = _notebook_tools()
    tickers = _check_tickers(prices, tickers, date_col=date_col)
    freq = _check_freq(freq)
    period = _check_period(period)

    supported_indicators = _available_indicators(
        prices,
        date_col=date_col,
    )
    requested_indicators = plot_kwargs.pop("indicators", None)

    if requested_indicators is None:
        available_indicators = supported_indicators
    else:
        available_indicators = _check_indicators(requested_indicators)
        unsupported = available_indicators - supported_indicators
        if unsupported:
            raise ValueError(
                "indicators unavailable for this price data: "
                f"{sorted(unsupported)}"
            )

<<<<<<< HEAD
    initial_visible = _check_indicators(
        plot_kwargs.pop(
            "visible_indicators",
            ("ma", "bb", "rsi"),
        )
    )
=======
    requested_visible = plot_kwargs.pop(
        "visible_indicators",
        None,
    )
    if requested_visible is None:
        initial_visible = {"ma", "volume"} & available_indicators
    else:
        initial_visible = _check_indicators(requested_visible)
>>>>>>> d1aad7a (candlestick)

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
        layout=widgets.Layout(width="auto"),
        style={"button_width": "auto"},
    )

    indicator_labels = {
        "ma": "MA",
        "bb": "BB",
        "rsi": "RSI",
        "ichimoku": "Ichimoku",
<<<<<<< HEAD
=======
        "atr": "ATR",
        "mfi": "MFI",
>>>>>>> d1aad7a (candlestick)
        "volume": "Volume",
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
            height=_CONTROL_HEIGHT,
            border="1px solid #ccc",
            padding="0 6px",
        ),
    )

    period_selector = widgets.ToggleButtons(
        options=_PERIOD_OPTIONS,
        value=period,
        description="",
        layout=widgets.Layout(width="auto"),
        style={"button_width": "auto"},
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
        layout=widgets.Layout(
            margin=group_margin,
        ),
    )
    period_group = widgets.Box(
        [period_selector],
    )

    controls = widgets.HBox(
        [
            ticker_group,
            freq_group,
            indicator_group,
            period_group,
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
                period=period_selector.value,
                indicators=selected_indicators,
                visible_indicators=selected_indicators,
                **plot_kwargs,
            ).show()

    ticker_selector.observe(render, names="value")
    freq_selector.observe(render, names="value")
    period_selector.observe(render, names="value")

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
    period: str = "ALL",
    base: float | None = 100.0,
    price_line_width: float = 2.0,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Compare tickers over a selected period from a common valid start date."""
    freq = _check_freq(freq)
    period = _check_period(period)
    tickers = _check_tickers(prices, tickers, date_col=date_col)
    _check_positive_number(price_line_width, "price_line_width")
    _check_figure_size(width, "width")
    _check_figure_size(height, "height")

    if base is not None:
        _check_positive_number(base, "base")

    data = _close_matrix(
        prices,
        tickers,
        date_col=date_col,
        freq=freq,
    )
    data = data.dropna(subset=tickers)
    if data.empty:
        raise ValueError("no common valid dates for requested tickers")

    data = _slice_period(data, period=period)
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


    return fig


def browse_price_comparison(
    prices: pd.DataFrame,
    tickers: Iterable[str],
    *,
    ticker_names: Mapping[str, str] | None = None,
    date_col: str = "date",
    normalize: bool = True,
    period: str = "ALL",
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
    When normalized, ``base`` is applied at the first common valid date
    within the selected period.
    """
    widgets, display, clear_output = _notebook_tools()
    tickers = _check_tickers(prices, tickers, date_col=date_col)
    period = _check_period(period)

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
        layout=widgets.Layout(width="auto"),
        style={"button_width": "auto"},
    )

    period_selector = widgets.ToggleButtons(
        options=_PERIOD_OPTIONS,
        value=period,
        description="",
        layout=widgets.Layout(width="auto"),
        style={"button_width": "auto"},
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
        layout=widgets.Layout(
            margin=group_margin,
        ),
    )

    period_group = widgets.Box(
        [period_selector],
    )

    controls = widgets.HBox(
        [
            ticker_group,
            normalize_group,
            period_group,
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
                period=period_selector.value,
                base=base if normalize_selector.value else None,
                price_line_width=price_line_width,
                width=width,
                height=height,
            ).show()

    ticker_selector.observe(render, names="value")
    normalize_selector.observe(render, names="value")
    period_selector.observe(render, names="value")
    render()

    display(widgets.VBox([controls, output]))


def _price_kind(
    prices: pd.DataFrame,
    *,
    date_col: str,
) -> str:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    if (
        isinstance(prices.index, pd.MultiIndex)
        and prices.index.nlevels == 2
        and set(_OHLCV_COLUMNS) <= set(prices.columns)
    ):
        return "ohlcv"

    if date_col in prices.columns:
        return "close"

    raise ValueError(
        "unsupported price format: expected close-wide data with "
        f"{date_col!r} plus ticker columns, or a 2-level (date, ticker) "
        "MultiIndex with open/high/low/close/volume columns"
    )


def _available_indicators(
    prices: pd.DataFrame,
    *,
    date_col: str,
) -> set[str]:
    kind = _price_kind(prices, date_col=date_col)

    if kind == "ohlcv":
        return set(_INDICATOR_ORDER)

    return {"ma", "bb", "rsi", "ichimoku"}


def _ticker_price_periods(
    prices: pd.DataFrame,
    ticker: str,
    *,
    date_col: str,
    freq: str,
) -> pd.DataFrame:
    kind = _price_kind(prices, date_col=date_col)

    if kind == "ohlcv":
        return _ohlcv_ticker_periods(
            prices,
            ticker,
            freq=freq,
        )

    close = _price_series(
        prices,
        ticker,
        date_col=date_col,
    )
    return _close_price_periods(close, freq=freq)


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


def _close_price_periods(
    close: pd.Series,
    *,
    freq: str,
) -> pd.DataFrame:
    if freq == "D":
        return pd.DataFrame(
            {
                "high": close,
                "low": close,
                "close": close,
            }
        )

    return close.resample("W-FRI").agg(
        high="max",
        low="min",
        close="last",
    ).dropna(subset=["close"])


def _ohlcv_ticker_periods(
    prices: pd.DataFrame,
    ticker: str,
    *,
    freq: str,
) -> pd.DataFrame:
    _check_tickers(prices, [ticker], date_col="date")

    data = prices.xs(ticker, level=1).copy()
    data = data.loc[:, list(_OHLCV_COLUMNS)]
    data.index = pd.to_datetime(data.index, errors="coerce")

    if data.index.isna().any():
        raise ValueError("OHLCV date index contains invalid dates")

    data = (
        data.loc[~data.index.duplicated(keep="last")]
        .sort_index()
        .apply(pd.to_numeric, errors="coerce")
    )
    data = data.dropna(subset=["open", "high", "low", "close"])

    if data.empty:
        raise ValueError(f"{ticker!r} has no valid OHLCV prices")

    price_cols = ["open", "high", "low", "close"]
    if (data[price_cols] <= 0).any().any():
        raise ValueError(f"{ticker!r} contains non-positive OHLC prices")
    if (data["volume"].dropna() < 0).any():
        raise ValueError(f"{ticker!r} contains negative volume")

    if freq == "D":
        return data

    return data.resample("W-FRI").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])


def _close_matrix(
    prices: pd.DataFrame,
    tickers: Sequence[str],
    *,
    date_col: str,
    freq: str,
) -> pd.DataFrame:
    kind = _price_kind(prices, date_col=date_col)

    if kind == "close":
        data = prices[[date_col, *tickers]].copy()
        data[date_col] = pd.to_datetime(
            data[date_col],
            errors="coerce",
        )
        if data[date_col].isna().any():
            raise ValueError(f"{date_col!r} contains invalid dates")

        data = (
            data.drop_duplicates(date_col, keep="last")
            .sort_values(date_col)
            .set_index(date_col)
        )
        data = data.apply(pd.to_numeric, errors="coerce")
    else:
        close = pd.to_numeric(prices["close"], errors="coerce")
        dates = pd.to_datetime(
            close.index.get_level_values(0),
            errors="coerce",
        )
        ticker_values = close.index.get_level_values(1)

        if dates.isna().any():
            raise ValueError("OHLCV date index contains invalid dates")

        tidy = pd.DataFrame(
            {
                date_col: dates,
                "_ticker": ticker_values,
                "_close": close.to_numpy(),
            }
        )
        data = tidy.pivot_table(
            index=date_col,
            columns="_ticker",
            values="_close",
            aggfunc="last",
        )
        data = data.reindex(columns=list(tickers)).sort_index()

    if freq == "W":
        data = data.resample("W-FRI").last()

    return data


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


<<<<<<< HEAD
=======

def _atr(
    data: pd.DataFrame,
    *,
    window: int,
) -> pd.Series:
    high = data["high"]
    low = data["low"]
    close = data["close"]
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window,
    ).mean()


def _mfi(
    data: pd.DataFrame,
    *,
    window: int,
) -> pd.Series:
    typical = (
        data["high"]
        + data["low"]
        + data["close"]
    ) / 3
    raw_flow = typical * data["volume"]

    direction = typical.diff()
    positive = raw_flow.where(direction > 0, 0.0)
    negative = raw_flow.where(direction < 0, 0.0)

    positive_sum = positive.rolling(window).sum()
    negative_sum = negative.rolling(window).sum()

    ratio = positive_sum / negative_sum
    mfi = 100 - 100 / (1 + ratio)
    mfi = mfi.mask(
        (negative_sum == 0) & (positive_sum > 0),
        100.0,
    )
    return mfi.mask(
        (negative_sum == 0) & (positive_sum == 0),
        50.0,
    )

>>>>>>> d1aad7a (candlestick)
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


def _check_period(period: str) -> str:
    period = str(period).upper()
    if period not in _PERIOD_OPTIONS:
        raise ValueError(
            f"period must be one of {list(_PERIOD_OPTIONS)}"
        )
    return period


def _period_start(
    end: pd.Timestamp,
    *,
    period: str,
) -> pd.Timestamp | None:
    period = _check_period(period)

    offsets = {
        "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6),
        "1Y": pd.DateOffset(years=1),
        "3Y": pd.DateOffset(years=3),
    }

    if period == "ALL":
        return None

    return end - offsets[period]


def _slice_period(
    data: pd.Series | pd.DataFrame,
    *,
    period: str,
) -> pd.Series | pd.DataFrame:
    if data.empty:
        return data

    start = _period_start(data.index.max(), period=period)
    if start is None:
        return data

    return data.loc[data.index >= start]


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
    kind = _price_kind(prices, date_col=date_col)

    if isinstance(tickers, str):
        result = [tickers]
    else:
        result = list(dict.fromkeys(tickers))

    if not result:
        raise ValueError("at least one ticker is required")
    if not all(isinstance(t, str) and t for t in result):
        raise ValueError("tickers must be non-empty strings")

    if kind == "close":
        available = set(prices.columns) - {date_col}
    else:
        available = set(
            prices.index.get_level_values(1).unique()
        )

    missing = [
        ticker
        for ticker in result
        if ticker not in available
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
