"""Reusable long-selection recipes and lightweight orchestration.

This module stores concrete combinations of selection policies. It also
re-exports the selection evaluation utility so routine use can rely on
``stock_assessment`` plus this module only.
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



__all__ = [
    "CORE_LONG_THRESHOLD_RULES",
    "CORE_LONG_TOP_N",
    "core_long_selection",
    "apply_selection",
    "evaluate_recipe",
    "evaluate_selection",
]


def core_long_selection() -> list[SelectionPolicy]:
    """Return the default reusable long-selection recipe.

    Step 1 applies loose threshold eligibility rules.
    Step 2 ranks the remaining universe with the default equal-group score
    and keeps the top ``CORE_LONG_TOP_N`` tickers.
    """

    return [
        LongThresholdFilter(),
        LongEqualGroupScore(
            top_n=CORE_LONG_TOP_N,
        ),
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
    prices: pd.DataFrame,
    ratios: pd.DataFrame,
    *,
    as_of,
    policies: Sequence[SelectionPolicy] | None = None,
    horizons: Iterable[int] = (1, 3, 6),
    tickers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Generate point-in-time features, apply a recipe, and evaluate returns.

    ``as_of`` is passed to ``assessment.assess`` so every selection feature is
    regenerated from information available at that point in time. Forward
    returns are then calculated only from prices after the resulting
    ``as_of_date``.
    """

    features = assessment.assess(
        prices,
        ratios,
        as_of=as_of,
    )

    selected = apply_selection(
        features,
        policies=policies,
        tickers=tickers,
    )

    try:
        price_date_col = (
            assessment.price_cols[
                "date"
            ]
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
    ) as exc:
        raise ValueError(
            "assessment must expose "
            "price_cols['date'] "
            "for recipe evaluation"
        ) from exc

    return evaluate_selection(
        selected,
        prices,
        horizons=horizons,
        result="detail",
        plot=False,
        ticker_col="ticker",
        as_of_col="as_of_date",
        price_date_col=price_date_col,
    )