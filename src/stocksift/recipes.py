"""Reusable long-selection recipes and lightweight orchestration.

This module stores concrete combinations of selection policies. It also
re-exports the selection evaluation utility so routine use can rely on
``assessment`` plus this module only.
"""

from __future__ import annotations


import warnings
from typing import Any, Iterable, Sequence

import pandas as pd
from tqdm.auto import tqdm

from .policy import (
    LongEqualGroupScore,
    LongThresholdFilter,
    SelectionPolicy,
    evaluate_selection,
)

CORE_LONG_TOP_N = 10


def core_long_selection() -> list[SelectionPolicy]:
    """Return the default reusable long-selection recipe.

    Step 1 applies loose threshold eligibility rules.
    Step 2 ranks the remaining universe with the default equal-group score & top_n
    """

    return [
        LongThresholdFilter(),
        LongEqualGroupScore(),
    ]


def apply_selection(
    features: pd.DataFrame,
    policies: Sequence[SelectionPolicy] | None = None,
    *,
    tickers: Iterable[str] | None = None,
    show_count: bool = True,
) -> pd.DataFrame:
    """Apply selection policies sequentially to one canonical feature table."""

    if policies is None:
        policies = core_long_selection()
    else:
        policies = list(policies)

    if not policies:
        raise ValueError(
            "at least one selection policy is required"
        )

    current_tickers = tickers
    result: pd.DataFrame | None = None

    initial_count = (
        len(features)
        if tickers is None
        else len(list(dict.fromkeys(tickers)))
    )

    counts = [initial_count]

    for step, policy in enumerate(
        policies,
        start=1,
    ):
        result = policy.select(
            features,
            tickers=current_tickers,
        )

        if result.empty:
            raise ValueError(
                "selection recipe produced no candidates "
                f"after step {step} "
                f"({policy.__class__.__name__})"
            )

        counts.append(len(result))

        current_tickers = result[
            policy.ticker_col
        ].tolist()

    assert result is not None

    if show_count:
        print(
            "Selected: "
            + " -> ".join(map(str, counts))
        )

    return result


def evaluate_recipe(
    assessment,
    *,
    as_of,
    policies: Sequence[SelectionPolicy] | None = None,
    horizons: Iterable[int] = (1, 3, 6),
    quantiles: int = 5,
    result: str = "detail",
    plot: str | bool = False,
    tickers: Iterable[str] | None = None,
) -> Any:
    """Assess, select, and evaluate a stock-selection recipe."""

    features = assessment.assess(
        as_of=as_of,
    )

    if policies is None:
        policies = core_long_selection()
    else:
        policies = list(policies)

    if tickers is None:
        universe = features
    else:
        requested = list(
            dict.fromkeys(tickers)
        )

        universe = features.loc[
            features["ticker"].isin(requested)
        ].copy()

    selected = apply_selection(
        features,
        policies=policies,
        tickers=tickers,
    )

    try:
        prices = assessment.prices
        price_date_col = assessment.price_cols["date"]
    except (
        AttributeError,
        KeyError,
        TypeError,
    ) as exc:
        raise ValueError(
            "assessment must expose loaded prices and "
            "price_cols['date'] for recipe evaluation"
        ) from exc

    if prices is None:
        raise ValueError(
            "assessment inputs are not loaded"
        )

    rank_col = getattr(
        policies[-1],
        "rank_col",
        "selection_rank",
    )

    return evaluate_selection(
        selected,
        prices,
        universe=universe,
        horizons=horizons,
        quantiles=quantiles,
        result=result,
        plot=plot,
        ticker_col="ticker",
        as_of_col="as_of_date",
        price_date_col=price_date_col,
        rank_col=rank_col,
    )



def selection_trajectory(
    assessment,
    *,
    as_of=None,
    policies: Sequence[SelectionPolicy] | None = None,
    top_n: int = CORE_LONG_TOP_N,
    freq: int = 5,
    periods: int = 5,
    style: bool = True,
) -> Any:
    """Review Top-N selection membership across historical assessment dates.

    Unlike evaluate_recipe(), which evaluates one point-in-time selection
    using future returns, selection_trajectory() compares the selections
    themselves across multiple historical as-of dates.

    ``freq`` is measured in available assessment dates rather than calendar
    days. The supplied policies are respected as-is; ``top_n`` only limits
    the number of rows displayed from each resulting selection.
    """
    if (
        isinstance(top_n, bool)
        or not isinstance(top_n, int)
        or top_n <= 0
    ):
        raise ValueError(
            "top_n must be a positive integer"
        )

    if (
        isinstance(freq, bool)
        or not isinstance(freq, int)
        or freq <= 0
    ):
        raise ValueError(
            "freq must be a positive integer"
        )

    if (
        isinstance(periods, bool)
        or not isinstance(periods, int)
        or periods <= 0
    ):
        raise ValueError(
            "periods must be a positive integer"
        )

    if policies is None:
        policies = core_long_selection()
    else:
        policies = list(policies)

    if not policies:
        raise ValueError(
            "at least one selection policy is required"
        )

    try:
        prices = assessment.prices
        ratios = assessment.ratios
        price_date_col = assessment.price_cols["date"]
        ratio_date_col = assessment.ratio_cols["date"]
    except (
        AttributeError,
        KeyError,
        TypeError,
    ) as exc:
        raise ValueError(
            "assessment must expose loaded prices, ratios, "
            "and configured date columns"
        ) from exc

    if prices is None or ratios is None:
        raise ValueError(
            "assessment inputs are not loaded"
        )

    price_dates = pd.to_datetime(
        prices[price_date_col],
        errors="raise",
    ).dropna()

    ratio_dates = pd.to_datetime(
        ratios[ratio_date_col],
        errors="raise",
    ).dropna()

    if price_dates.empty or ratio_dates.empty:
        raise ValueError(
            "assessment inputs contain no usable dates"
        )

    latest_common_date = min(
        price_dates.max(),
        ratio_dates.max(),
    )

    requested_as_of = (
        latest_common_date
        if as_of is None
        else min(
            pd.Timestamp(as_of),
            latest_common_date,
        )
    )

    available_dates = (
        ratio_dates.loc[
            ratio_dates <= requested_as_of
        ]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if available_dates.empty:
        raise ValueError(
            "no assessment date is available on or before as_of"
        )

    positions = [
        len(available_dates) - 1 - i * freq
        for i in range(periods)
        if len(available_dates) - 1 - i * freq >= 0
    ]

    as_of_dates = (
        available_dates.iloc[
            list(reversed(positions))
        ]
        .tolist()
    )

    if len(as_of_dates) < periods:
        warnings.warn(
            f"Only {len(as_of_dates)} trajectory dates are available "
            f"for periods={periods} and freq={freq}.",
            stacklevel=2,
        )

    ticker_col = policies[-1].ticker_col
    rank_col = getattr(
        policies[-1],
        "rank_col",
        None,
    )

    if rank_col is None:
        raise ValueError(
            "the final selection policy must produce ranked results"
        )

    selections: dict[str, list[str]] = {}
    short_counts: list[int] = []

    for date in tqdm(
        as_of_dates,
        desc="Selection trajectory",
    ):
        features = assessment.assess(
            as_of=date,
        )

        selected = apply_selection(
            features,
            policies=policies,
            show_count=False,
        )

        if rank_col not in selected.columns:
            raise ValueError(
                f"selection result is missing rank column {rank_col!r}"
            )

        selected = (
            selected
            .sort_values(rank_col)
            .head(top_n)
        )

        actual_as_of = str(
            features["as_of_date"].iloc[0]
        )

        tickers = (
            selected[ticker_col]
            .astype(str)
            .tolist()
        )

        selections[actual_as_of] = tickers

        if len(tickers) < top_n:
            short_counts.append(
                len(tickers)
            )

    if short_counts:
        warnings.warn(
            f"Some selections contain fewer than trajectory "
            f"top_n={top_n}; minimum selection size="
            f"{min(short_counts)}. "
            "Policy results were kept unchanged.",
            stacklevel=2,
        )

    max_rows = min(
        top_n,
        max(
            len(tickers)
            for tickers in selections.values()
        ),
    )

    trajectory = pd.DataFrame(
        {
            date: (
                tickers
                + [pd.NA] * (
                    max_rows - len(tickers)
                )
            )[:max_rows]
            for date, tickers in selections.items()
        },
        index=pd.RangeIndex(
            1,
            max_rows + 1,
            name="rank",
        ),
    )

    if not style:
        return trajectory
    
    # Only tickers selected on the latest assessment date are highlighted.
    latest_tickers = [
        ticker
        for ticker in trajectory.iloc[:, -1]
        if pd.notna(ticker)
    ]
    
    color_map = {
        ticker: (
            f"hsl("
            f"{round(i * 360 / max(len(latest_tickers), 1))}, "
            f"55%, 85%)"
        )
        for i, ticker in enumerate(latest_tickers)
    }
    
    def _cell_style(value):
        if pd.isna(value) or value not in color_map:
            return ""
    
        return (
            "background-color: "
            f"{color_map[value]}; "
            "text-align: center"
        )
    
    styler = (
        trajectory.style
        .map(_cell_style)
        .set_properties(
            **{
                "text-align": "center",
            }
        )
    )
    
    ticker_names = getattr(
        assessment,
        "ticker_names",
        None,
    )
    
    if ticker_names is None:
        return styler
    
    def _ticker_name(ticker):
        if pd.isna(ticker):
            return ""
    
        name = ticker_names.get(
            ticker,
            "",
        )
    
        return (
            ""
            if pd.isna(name)
            else str(name)
        )
    
    tooltips = trajectory.apply(
        lambda column: column.map(
            _ticker_name
        )
    )
    
    return styler.set_tooltips(
        tooltips
    )