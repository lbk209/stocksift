"""Reusable long-selection recipes and lightweight orchestration.

This module stores concrete combinations of selection policies. It also
re-exports the selection evaluation utility so routine use can rely on
``assessment`` plus this module only.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from .policy import (
    LongEqualGroupScore,
    LongThresholdFilter,
    SelectionPolicy,
    evaluate_selection,
)


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