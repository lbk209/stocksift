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
    CORE_LONG_TOP_N,
    DEFAULT_LONG_SCORE_GROUPS,
    LongEqualGroupScore,
    LongGroupScore,
    LongThresholdFilter,
    SelectionPolicy,
    evaluate_selection,
)


# Initial filter presets for notebook validation. Strict intentionally keeps
# the previous core filter thresholds, while Mild relaxes each tail guardrail.
MILD_LONG_THRESHOLD_RULES = (
    ("per_hist_pct", "<=", 90.0),
    ("eps_growth_pct", ">=", 10.0),
    ("momentum_12m_pct", ">=", 10.0),
)

STRICT_LONG_THRESHOLD_RULES = (
    ("per_hist_pct", "<=", 80.0),
    ("eps_growth_pct", ">=", 20.0),
    ("momentum_12m_pct", ">=", 20.0),
)


# Group weights are intentionally distinct enough for the named strategies to
# represent different ranking preferences rather than minor tuning variants.
VALUE_LONG_GROUP_WEIGHTS = {
    "valuation": 0.50,
    "fundamentals": 0.30,
    "momentum": 0.20,
}

MOMENTUM_LONG_GROUP_WEIGHTS = {
    "valuation": 0.20,
    "fundamentals": 0.20,
    "momentum": 0.60,
}

FUNDAMENTAL_LONG_GROUP_WEIGHTS = {
    "valuation": 0.25,
    "fundamentals": 0.55,
    "momentum": 0.20,
}

VALUE_MOMENTUM_LONG_GROUP_WEIGHTS = {
    "valuation": 0.40,
    "fundamentals": 0.20,
    "momentum": 0.40,
}


def _weighted_group_rules() -> dict[str, dict[str, dict[str, object]]]:
    """Convert the default group directions to LongGroupScore rule format."""
    return {
        group: {
            feature: {
                "direction": direction,
                "weight": 1.0,
            }
            for feature, direction in feature_directions.items()
        }
        for group, feature_directions in DEFAULT_LONG_SCORE_GROUPS.items()
    }


def mild_long_filter() -> LongThresholdFilter:
    """Return the loose long eligibility filter preset."""
    return LongThresholdFilter(
        rules=MILD_LONG_THRESHOLD_RULES,
    )


def strict_long_filter() -> LongThresholdFilter:
    """Return the stricter long eligibility filter preset."""
    return LongThresholdFilter(
        rules=STRICT_LONG_THRESHOLD_RULES,
    )


def balanced_long_strategy(
    *,
    top_n: int | None = CORE_LONG_TOP_N,
) -> LongEqualGroupScore:
    """Rank valuation, fundamentals, and momentum with equal group weights."""
    return LongEqualGroupScore(
        top_n=top_n,
    )


def _weighted_long_strategy(
    group_weights: dict[str, float],
    *,
    top_n: int | None,
) -> LongGroupScore:
    """Build a weighted-group long ranking strategy."""
    return LongGroupScore(
        rules=_weighted_group_rules(),
        group_weights=group_weights,
        top_n=top_n,
    )


def value_long_strategy(
    *,
    top_n: int | None = CORE_LONG_TOP_N,
) -> LongGroupScore:
    """Emphasize valuation while retaining fundamentals and momentum."""
    return _weighted_long_strategy(
        VALUE_LONG_GROUP_WEIGHTS,
        top_n=top_n,
    )


def momentum_long_strategy(
    *,
    top_n: int | None = CORE_LONG_TOP_N,
) -> LongGroupScore:
    """Emphasize momentum while retaining valuation and fundamentals."""
    return _weighted_long_strategy(
        MOMENTUM_LONG_GROUP_WEIGHTS,
        top_n=top_n,
    )


def fundamental_long_strategy(
    *,
    top_n: int | None = CORE_LONG_TOP_N,
) -> LongGroupScore:
    """Emphasize fundamentals while retaining valuation and momentum."""
    return _weighted_long_strategy(
        FUNDAMENTAL_LONG_GROUP_WEIGHTS,
        top_n=top_n,
    )


def value_momentum_long_strategy(
    *,
    top_n: int | None = CORE_LONG_TOP_N,
) -> LongGroupScore:
    """Jointly emphasize valuation and momentum."""
    return _weighted_long_strategy(
        VALUE_MOMENTUM_LONG_GROUP_WEIGHTS,
        top_n=top_n,
    )


LONG_FILTER_REGISTRY = {
    "mild": {
        "label": "Mild",
        "desc": "Loose tail guardrails before ranking.",
        "factory": mild_long_filter,
    },
    "strict": {
        "label": "Strict",
        "desc": "Stronger tail guardrails before ranking.",
        "factory": strict_long_filter,
    },
}

LONG_STRATEGY_REGISTRY = {
    "balanced": {
        "label": "Balanced",
        "desc": "Equal emphasis on valuation, fundamentals, and momentum.",
        "factory": balanced_long_strategy,
    },
    "value": {
        "label": "Value",
        "desc": "Emphasizes valuation while retaining other groups.",
        "factory": value_long_strategy,
    },
    "momentum": {
        "label": "Momentum",
        "desc": "Emphasizes momentum while retaining other groups.",
        "factory": momentum_long_strategy,
    },
    "fundamental": {
        "label": "Fundamental",
        "desc": "Emphasizes fundamentals while retaining other groups.",
        "factory": fundamental_long_strategy,
    },
    "value_momentum": {
        "label": "Value + Momentum",
        "desc": "Jointly emphasizes valuation and momentum.",
        "factory": value_momentum_long_strategy,
    },
}


def build_long_selection(
    *,
    filter: str = "mild",
    strategy: str = "balanced",
    top_n: int | None = CORE_LONG_TOP_N,
) -> list[SelectionPolicy]:
    """Build a long-selection pipeline from filter and strategy presets."""
    if filter not in LONG_FILTER_REGISTRY:
        raise ValueError(
            f"unknown long filter {filter!r}; "
            f"use one of {list(LONG_FILTER_REGISTRY)}"
        )

    if strategy not in LONG_STRATEGY_REGISTRY:
        raise ValueError(
            f"unknown long strategy {strategy!r}; "
            f"use one of {list(LONG_STRATEGY_REGISTRY)}"
        )

    filter_factory = LONG_FILTER_REGISTRY[filter]["factory"]
    strategy_factory = LONG_STRATEGY_REGISTRY[strategy]["factory"]

    return [
        filter_factory(),
        strategy_factory(top_n=top_n),
    ]


def default_long_selection(
    *,
    top_n: int | None = CORE_LONG_TOP_N,
) -> list[SelectionPolicy]:
    """Return the default Mild + Balanced long-selection pipeline."""
    return build_long_selection(
        filter="mild",
        strategy="balanced",
        top_n=top_n,
    )


def _completed_months(start, end) -> int:
    """Return completed calendar months between two timestamps."""
    months = (
        (end.year - start.year) * 12
        + end.month
        - start.month
        - (end.day < start.day)
    )
    return max(0, months)


def _analysis_history(
    assessment,
) -> tuple[int, pd.Timestamp] | None:
    """Return usable analysis months and earliest usable assessment date."""
    try:
        prices = assessment.prices
        ratios = assessment.ratios
        price_date_col = assessment.price_cols["date"]
        ratio_date_col = assessment.ratio_cols["date"]
    except (AttributeError, KeyError, TypeError):
        return None

    if prices is None or ratios is None:
        return None

    price_dates = pd.to_datetime(
        prices[price_date_col],
        errors="raise",
    ).dropna()
    ratio_dates = pd.to_datetime(
        ratios[ratio_date_col],
        errors="raise",
    ).dropna()

    if price_dates.empty or ratio_dates.empty:
        return None

    data_start = max(
        price_dates.min(),
        ratio_dates.min(),
    )
    data_end = min(
        price_dates.max(),
        ratio_dates.max(),
    )
    required_months = getattr(
        assessment,
        "RECOMMENDED_HISTORY_MONTHS",
        13,
    )
    earliest_usable_as_of = (
        data_start
        + pd.DateOffset(months=required_months)
    )
    available_months = (
        _completed_months(
            earliest_usable_as_of,
            data_end,
        )
        if data_end >= earliest_usable_as_of
        else 0
    )

    return available_months, earliest_usable_as_of


def apply_selection(
    features: pd.DataFrame,
    policies: Sequence[SelectionPolicy] | None = None,
    *,
    tickers: Iterable[str] | None = None,
    show_count: bool = True,
) -> pd.DataFrame:
    """Apply selection policies sequentially to one canonical feature table."""

    if policies is None:
        policies = default_long_selection()
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

    horizons = tuple(horizons)

    features = assessment.assess(
        as_of=as_of,
    )

    analysis_history = _analysis_history(assessment)
    if analysis_history is not None:
        available_months, _ = analysis_history
        unavailable_horizons = [
            horizon
            for horizon in horizons
            if (
                isinstance(horizon, int)
                and not isinstance(horizon, bool)
                and horizon > available_months
            )
        ]

        if unavailable_horizons:
            warnings.warn(
                f"Requested horizons {unavailable_horizons}M exceed the "
                f"approximately {available_months}-month usable "
                "historical analysis window.",
                stacklevel=2,
            )

    if policies is None:
        policies = default_long_selection()
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
    top_n: int | None = None,
    freq: int = 5,
    periods: int = 5,
    style: bool = True,
) -> Any:
    """Review Top-N selection membership across historical assessment dates.

    Unlike evaluate_recipe(), which evaluates one point-in-time selection
    using future returns, selection_trajectory() compares the selections
    themselves across multiple historical as-of dates.

    ``freq`` is measured in available assessment dates rather than calendar
    days. The supplied policies are respected as-is. ``top_n`` optionally
    limits the number of rows displayed from each resulting selection; when
    ``None``, the policy result is kept unchanged.
    """
    if top_n is not None and (
        isinstance(top_n, bool)
        or not isinstance(top_n, int)
        or top_n <= 0
    ):
        raise ValueError(
            "top_n must be a positive integer or None"
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
        policies = default_long_selection()
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

    analysis_history = _analysis_history(assessment)
    trajectory_start = pd.Timestamp(as_of_dates[0])

    if analysis_history is not None:
        _, earliest_usable_as_of = analysis_history

        if trajectory_start < earliest_usable_as_of:
            warnings.warn(
                f"Requested trajectory starts on "
                f"{trajectory_start.date().isoformat()}, before the "
                f"earliest approximately usable assessment date "
                f"{earliest_usable_as_of.date().isoformat()}; "
                "results may be incomplete.",
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
            print_msg=False,
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

        selected = selected.sort_values(rank_col)
        if top_n is not None:
            selected = selected.head(top_n)

        actual_as_of = str(
            features["as_of_date"].iloc[0]
        )

        tickers = (
            selected[ticker_col]
            .astype(str)
            .tolist()
        )

        selections[actual_as_of] = tickers

        if top_n is not None and len(tickers) < top_n:
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

    max_rows = max(
        len(tickers)
        for tickers in selections.values()
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

    min_lightness = 70
    max_lightness = 95
    color_map = {
        ticker: (
            f"hsl("
            #f"{round(i * 360 / max(len(latest_tickers), 1))}, "
            f"{round(i * 180 / max(len(latest_tickers) - 1, 1))}, "
            f"55% ,"
            f"{round(
                min_lightness
                + i * (max_lightness - min_lightness)
                / max(len(latest_tickers) - 1, 1)
            )}%)"
            #85%)"
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
