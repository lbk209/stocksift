"""Long-only stock-selection policies for ticker-level feature data.

The module consumes only an already-computed ticker-level feature DataFrame.
It does not read StockAssessment configuration or depend on StockAssessment.

All concrete selection classes use `rules` for policy configuration and
`features` for the ticker-level DataFrame passed to select().

Every select() result preserves the selected rows' original feature columns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import operator
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ThresholdRule = tuple[str, str, float]


CORE_LONG_TOP_N = 10


# Loose eligibility guardrails remove clearly unattractive tails before the
# equal-group score ranks valuation, fundamentals, and momentum. They are
# intentionally simple rather than optimized to historical results.
CORE_LONG_THRESHOLD_RULES = (
    ("per_hist_pct", "<=", 80.0),
    ("eps_growth_pct", ">=", 20.0),
    ("momentum_12m_pct", ">=", 20.0),
)


# Default normalized features used by the equal-feature long score.
#
# Feature choice belongs to the selection policy and is independent of
# StockAssessment configuration.
#
# Risk features are intentionally excluded from the default long score.
# They remain available in the feature DataFrame for separate review.
DEFAULT_LONG_SCORE_FEATURES = {
    # Valuation: lower is more attractive.
    #"per_hist_pct": "lower",
    #"pbr_hist_pct": "lower",
    "per_expansion_pct": "lower",
    "pbr_expansion_pct": "lower",

    # Fundamentals: higher growth is more attractive.
    "eps_growth_pct": "higher",
    "bps_growth_pct": "higher",
    #"dps_growth_pct": "higher",

    # Momentum: higher relative momentum is more attractive.
    "momentum_6m_pct": "higher",
    "momentum_12m_pct": "higher",
}


# Default groups used by the equal-group long score.
DEFAULT_LONG_SCORE_GROUPS = {
    "valuation": {
        "per_hist_pct": "lower",
        "pbr_hist_pct": "lower",
        "per_expansion_pct": "lower",
        "pbr_expansion_pct": "lower",
    },
    "fundamentals": {
        "eps_growth_pct": "higher",
        "bps_growth_pct": "higher",
        #"dps_growth_pct": "higher",
    },
    "momentum": {
        "momentum_6m_pct": "higher",
        "momentum_12m_pct": "higher",
    },
}


@dataclass(frozen=True)
class _FeatureSpec:
    """Normalized scoring specification for one 0-100 feature."""

    direction: str
    weight: float


class SelectionPolicy(ABC):
    """Minimal interface shared by selection policies."""

    def __init__(self, *, ticker_col: str = "ticker") -> None:
        self.ticker_col = ticker_col

    @abstractmethod
    def select(
        self,
        features: pd.DataFrame,
        *,
        tickers: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Select or rank rows within an optional ticker subset."""

    def _prepare_input(
        self,
        features: pd.DataFrame,
        *,
        tickers: Iterable[str] | None,
        required_columns: Iterable[str] = (),
    ) -> pd.DataFrame:
        """Validate the feature table and optionally restrict its tickers."""

        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")

        if self.ticker_col not in features.columns:
            raise ValueError(
                f"features must contain ticker column {self.ticker_col!r}"
            )

        if features[self.ticker_col].duplicated().any():
            raise ValueError(
                "features must contain at most one row per ticker"
            )

        missing_columns = sorted(
            set(required_columns).difference(features.columns)
        )
        if missing_columns:
            raise ValueError(
                f"features is missing policy columns: {missing_columns}"
            )

        data = features.copy()

        if tickers is None:
            return data

        if isinstance(tickers, str):
            requested = [tickers]
        else:
            requested = list(dict.fromkeys(tickers))

        available = set(data[self.ticker_col])

        missing_tickers = [
            ticker
            for ticker in requested
            if ticker not in available
        ]

        if missing_tickers:
            suffix = " ..." if len(missing_tickers) > 10 else ""
            raise ValueError(
                "requested tickers are missing from features: "
                f"{missing_tickers[:10]}{suffix}"
            )

        requested_set = set(requested)

        return data.loc[
            data[self.ticker_col].isin(requested_set)
        ].copy()


class LongThresholdFilter(SelectionPolicy):
    """Filter long candidates with fixed threshold rules and AND logic.

    `rules` is a sequence of:

        (feature, operator, threshold)

    Example:

        rules=[
            ("per_hist_pct", "<=", 50),
            ("eps_growth_pct", ">=", 50),
            ("momentum_12m_pct", ">=", 50),
        ]

    All rules must pass. Feature values are used exactly as supplied;
    ranks or percentiles are not recalculated after ticker filtering.

    Missing values fail the corresponding rule.

    Passing rows are not ranked.
    """

    _OPERATORS = {
        "<=": operator.le,
        "<": operator.lt,
        ">=": operator.ge,
        ">": operator.gt,
    }

    def __init__(
        self,
        rules: Sequence[ThresholdRule] | None = None,
    ) -> None:
        if rules is None:
            rules = CORE_LONG_THRESHOLD_RULES

        super().__init__()
        self.rules = self._normalize_rules(rules)

    def select(
        self,
        features: pd.DataFrame,
        *,
        tickers: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Filter long candidates from ticker-level features.

        Example input:

            features = pd.DataFrame({
                "ticker": ["005930", "000660", "035420"],
                "per_hist_pct": [35.0, 72.0, 48.0],
                "eps_growth_pct": [65.0, 80.0, 42.0],
                "momentum_12m_pct": [70.0, 91.0, 55.0],
            })

        `tickers` optionally restricts the rows evaluated without
        recalculating feature values or percentiles.
        """

        data = self._prepare_input(
            features,
            tickers=tickers,
            required_columns=(
                feature
                for feature, _, _ in self.rules
            ),
        )

        if not self.rules:
            return data.reset_index(drop=True)

        mask = pd.Series(True, index=data.index)

        for feature, op, threshold in self.rules:
            mask &= self._OPERATORS[op](
                data[feature],
                threshold,
            ).fillna(False)

        return (
            data.loc[mask]
            .copy()
            .reset_index(drop=True)
        )

    def trace(
        self,
        features: pd.DataFrame,
        *,
        tickers: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Show the remaining ticker count after each threshold rule."""

        remaining = self._prepare_input(
            features,
            tickers=tickers,
            required_columns=(
                feature
                for feature, _, _ in self.rules
            ),
        )

        rows = [
            {
                "step": 0,
                "feature": "initial_universe",
                "operator": "",
                "threshold": pd.NA,
                "remaining": len(remaining),
            }
        ]

        for step, (feature, op, threshold) in enumerate(
            self.rules,
            start=1,
        ):
            passed = self._OPERATORS[op](
                remaining[feature],
                threshold,
            ).fillna(False)

            remaining = remaining.loc[passed]

            rows.append(
                {
                    "step": step,
                    "feature": feature,
                    "operator": op,
                    "threshold": threshold,
                    "remaining": len(remaining),
                }
            )

        return pd.DataFrame(rows)

    def _normalize_rules(
        self,
        rules: Sequence[ThresholdRule],
    ) -> tuple[ThresholdRule, ...]:
        normalized: list[ThresholdRule] = []

        for rule in rules:
            if len(rule) != 3:
                raise ValueError(
                    "each threshold rule must be "
                    "(feature, operator, threshold)"
                )

            feature, op, threshold = rule

            if not isinstance(feature, str) or not feature:
                raise ValueError(
                    "rule feature must be a non-empty string"
                )

            if op not in self._OPERATORS:
                raise ValueError(
                    f"unsupported operator {op!r}; "
                    f"use one of {sorted(self._OPERATORS)}"
                )

            if not isinstance(threshold, (int, float)):
                raise TypeError(
                    "rule threshold must be numeric"
                )

            normalized.append(
                (feature, op, float(threshold))
            )

        return tuple(normalized)


class _ScoreSelection(SelectionPolicy, ABC):
    """Shared validation, scoring, ranking, and top-N mechanics."""

    def __init__(
        self,
        *,
        top_n: int | None = CORE_LONG_TOP_N,
        score_col: str = "selection_score",
        rank_col: str = "selection_rank",
        ticker_col: str = "ticker",
    ) -> None:
        super().__init__(ticker_col=ticker_col)

        if top_n is not None and (
            not isinstance(top_n, int)
            or top_n <= 0
        ):
            raise ValueError(
                "top_n must be a positive integer or None"
            )

        self.top_n = top_n
        self.score_col = score_col
        self.rank_col = rank_col

    def select(
        self,
        features: pd.DataFrame,
        *,
        tickers: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Calculate long scores and rank the selected ticker universe."""

        required = self._required_features()

        data = self._prepare_input(
            features,
            tickers=tickers,
            required_columns=required,
        )

        self._validate_score_features(
            data,
            required,
        )

        # Diagnose required scoring features that contain no usable values
        # anywhere in the current ticker universe.
        all_missing = [
            feature
            for feature in required
            if pd.to_numeric(
                data[feature],
                errors="coerce",
            ).notna().sum() == 0
        ]

        if all_missing:
            raise ValueError(
                "selection cannot be scored because required features "
                f"contain no usable values: {all_missing}"
            )

        score, extra_columns = self._calculate_score(data)

        output_names = [
            self.score_col,
            self.rank_col,
            *extra_columns,
        ]

        collisions = sorted(
            set(output_names).intersection(data.columns)
        )

        if collisions:
            raise ValueError(
                "features already contains selection output columns: "
                f"{collisions}. "
                "Use the intended feature table as input."
            )

        result = data.copy()

        for name, values in extra_columns.items():
            result[name] = values

        result[self.score_col] = score

        # All configured scoring inputs must be available for a ticker.
        # Missing inputs are not handled by dynamic weight redistribution.
        valid_score = result[self.score_col].notna()

        if not valid_score.any():
            raise ValueError(
                "selection cannot be scored because no ticker has "
                "a complete set of required scoring features"
            )

        result = result.loc[valid_score].copy()

        result = result.sort_values(
            [self.score_col, self.ticker_col],
            ascending=[False, True],
            kind="stable",
        )

        result[self.rank_col] = np.arange(
            1,
            len(result) + 1,
        )

        if self.top_n is not None:
            result = result.head(self.top_n)

        return result.reset_index(drop=True)

    @abstractmethod
    def _required_features(self) -> tuple[str, ...]:
        """Return feature columns required to calculate the score."""

    @abstractmethod
    def _calculate_score(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.Series, dict[str, pd.Series]]:
        """Return final score and optional intermediate score columns."""

    @staticmethod
    def _normalize_feature_rules(
        rules: Mapping[str, Mapping[str, object]],
        *,
        default_weight: float,
    ) -> dict[str, _FeatureSpec]:
        """Normalize weighted feature rules into internal feature specs."""

        if not rules:
            raise ValueError(
                "at least one scoring feature is required"
            )

        if (
            not isinstance(default_weight, (int, float))
            or default_weight <= 0
        ):
            raise ValueError(
                "default_weight must be positive"
            )

        normalized: dict[str, _FeatureSpec] = {}

        for feature, rule in rules.items():
            if not isinstance(feature, str) or not feature:
                raise ValueError(
                    "feature names must be non-empty strings"
                )

            if not isinstance(rule, Mapping):
                raise TypeError(
                    f"rule for {feature!r} must be a mapping"
                )

            direction = rule.get("direction")
            weight = rule.get(
                "weight",
                default_weight,
            )

            if direction not in {"higher", "lower"}:
                raise ValueError(
                    f"direction for {feature!r} must be "
                    "'higher' or 'lower'"
                )

            if (
                not isinstance(weight, (int, float))
                or weight <= 0
            ):
                raise ValueError(
                    f"weight for {feature!r} must be positive"
                )

            normalized[feature] = _FeatureSpec(
                direction=str(direction),
                weight=float(weight),
            )

        return normalized

    @staticmethod
    def _validate_score_features(
        data: pd.DataFrame,
        feature_names: Iterable[str],
    ) -> None:
        """Validate that scoring features lie within the 0-100 range."""

        for feature in feature_names:
            values = pd.to_numeric(
                data[feature],
                errors="coerce",
            )

            non_null = values.dropna()

            if non_null.empty:
                continue

            if (
                (non_null < 0)
                | (non_null > 100)
            ).any():
                raise ValueError(
                    f"scoring feature {feature!r} "
                    "must be normalized to 0-100"
                )

    @staticmethod
    def _feature_score(
        data: pd.DataFrame,
        rules: Mapping[str, _FeatureSpec],
    ) -> pd.Series:
        """Convert feature direction and calculate a weighted score."""

        desirability = pd.DataFrame(
            index=data.index
        )

        weights: dict[str, float] = {}

        for feature, rule in rules.items():
            values = pd.to_numeric(
                data[feature],
                errors="coerce",
            )

            desirability[feature] = (
                values
                if rule.direction == "higher"
                else 100.0 - values
            )

            weights[feature] = rule.weight

        return _ScoreSelection._strict_weighted_mean(
            desirability,
            weights,
        )

    @staticmethod
    def _strict_weighted_mean(
        values: pd.DataFrame,
        weights: Mapping[str, float],
    ) -> pd.Series:
        """Calculate a weighted mean only when all inputs are available."""

        ordered_weights = pd.Series(
            {
                column: float(weights[column])
                for column in values.columns
            },
            dtype=float,
        )

        if (ordered_weights <= 0).any():
            raise ValueError(
                "all weights must be positive"
            )

        complete = values.notna().all(axis=1)

        score = values.mul(
            ordered_weights,
            axis="columns",
        ).sum(axis=1)

        score = score / ordered_weights.sum()

        return score.where(complete)


class LongFeatureScore(_ScoreSelection):
    """Rank long candidates by a weighted mean of normalized features.

    `rules` maps each scoring feature to its long-score direction and
    optional weight.

    Example:

        rules={
            "per_hist_pct": {
                "direction": "lower",
                "weight": 1.0,
            },
            "eps_growth_pct": {
                "direction": "higher",
                "weight": 2.0,
            },
            "momentum_12m_pct": {
                "direction": "higher",
                "weight": 1.0,
            },
        }

    Scoring features must already be normalized to the 0-100 range.
    """

    DEFAULT_WEIGHT = 1.0

    def __init__(
        self,
        rules: Mapping[str, Mapping[str, object]],
        *,
        top_n: int | None = CORE_LONG_TOP_N,
    ) -> None:
        super().__init__(
            top_n=top_n,
        )

        self.rules = self._normalize_feature_rules(
            rules,
            default_weight=self.DEFAULT_WEIGHT,
        )

    def _required_features(self) -> tuple[str, ...]:
        return tuple(self.rules)

    def _calculate_score(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.Series, dict[str, pd.Series]]:
        return (
            self._feature_score(
                data,
                self.rules,
            ),
            {},
        )


class LongEqualFeatureScore(LongFeatureScore):
    """Equal-weight convenience form of LongFeatureScore.

    `rules` maps each scoring feature directly to its long-score direction.

    Example:

        rules={
            "per_hist_pct": "lower",
            "eps_growth_pct": "higher",
            "momentum_12m_pct": "higher",
        }

    If `rules` is omitted, DEFAULT_LONG_SCORE_FEATURES is used.

    Every configured feature is explicitly assigned weight 1.0.
    """

    EQUAL_WEIGHT = 1.0

    def __init__(
        self,
        rules: Mapping[str, str] | None = None,
        *,
        top_n: int | None = CORE_LONG_TOP_N,
    ) -> None:
        if rules is None:
            rules = DEFAULT_LONG_SCORE_FEATURES

        # Equal weighting is explicit and does not depend on
        # LongFeatureScore.DEFAULT_WEIGHT.
        equal_rules = {
            feature: {
                "direction": direction,
                "weight": self.EQUAL_WEIGHT,
            }
            for feature, direction in rules.items()
        }

        super().__init__(
            equal_rules,
            top_n=top_n,
        )


class LongGroupScore(_ScoreSelection):
    """Rank long candidates using weighted features inside weighted groups.

    `rules` maps each group to weighted feature rules.

    Example:

        rules={
            "valuation": {
                "per_hist_pct": {
                    "direction": "lower",
                    "weight": 2.0,
                },
                "pbr_hist_pct": {
                    "direction": "lower",
                    "weight": 1.0,
                },
            },
            "momentum": {
                "momentum_6m_pct": {
                    "direction": "higher",
                    "weight": 1.0,
                },
                "momentum_12m_pct": {
                    "direction": "higher",
                    "weight": 2.0,
                },
            },
        }

        group_weights={
            "valuation": 1.0,
            "momentum": 1.5,
        }

    Each group score is the weighted mean of its features.
    The final selection score is the weighted mean of the group scores.
    """

    DEFAULT_FEATURE_WEIGHT = 1.0
    DEFAULT_GROUP_WEIGHT = 1.0

    def __init__(
        self,
        rules: Mapping[
            str,
            Mapping[str, Mapping[str, object]],
        ],
        *,
        group_weights: Mapping[str, float] | None = None,
        top_n: int | None = CORE_LONG_TOP_N,
    ) -> None:
        super().__init__(
            top_n=top_n,
        )

        if not rules:
            raise ValueError(
                "at least one scoring group is required"
            )

        self.rules: dict[
            str,
            dict[str, _FeatureSpec],
        ] = {}

        for group, feature_rules in rules.items():
            if not isinstance(group, str) or not group:
                raise ValueError(
                    "group names must be non-empty strings"
                )

            self.rules[group] = (
                self._normalize_feature_rules(
                    feature_rules,
                    default_weight=self.DEFAULT_FEATURE_WEIGHT,
                )
            )

        self.group_weights = (
            self._normalize_group_weights(
                group_weights
            )
        )

    def _normalize_group_weights(
        self,
        group_weights: Mapping[str, float] | None,
    ) -> dict[str, float]:
        if group_weights is None:
            weights = {
                group: float(
                    self.DEFAULT_GROUP_WEIGHT
                )
                for group in self.rules
            }
        else:
            missing = sorted(
                set(self.rules).difference(
                    group_weights
                )
            )

            extra = sorted(
                set(group_weights).difference(
                    self.rules
                )
            )

            if missing or extra:
                raise ValueError(
                    "group_weights keys must exactly match rules; "
                    f"missing={missing}, extra={extra}"
                )

            weights = {
                group: float(weight)
                for group, weight
                in group_weights.items()
            }

        if any(
            weight <= 0
            for weight in weights.values()
        ):
            raise ValueError(
                "all group weights must be positive"
            )

        return weights

    def _required_features(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                feature
                for feature_rules in self.rules.values()
                for feature in feature_rules
            )
        )

    def _calculate_score(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.Series, dict[str, pd.Series]]:
        group_scores = pd.DataFrame(
            index=data.index
        )

        extra_columns: dict[str, pd.Series] = {}

        for group, feature_rules in self.rules.items():
            score = self._feature_score(
                data,
                feature_rules,
            )

            group_scores[group] = score
            extra_columns[f"{group}_score"] = score

        final_score = self._strict_weighted_mean(
            group_scores,
            self.group_weights,
        )

        return final_score, extra_columns


class LongEqualGroupScore(LongGroupScore):
    """Equal-weight convenience form of LongGroupScore.

    `rules` maps each group to feature directions without requiring
    explicit feature or group weights.

    Example:

        rules={
            "valuation": {
                "per_hist_pct": "lower",
                "pbr_hist_pct": "lower",
            },
            "fundamentals": {
                "eps_growth_pct": "higher",
                "bps_growth_pct": "higher",
            },
            "momentum": {
                "momentum_6m_pct": "higher",
                "momentum_12m_pct": "higher",
            },
        }

    If `rules` is omitted, DEFAULT_LONG_SCORE_GROUPS is used.

    Every feature within each group is explicitly assigned weight 1.0,
    and every group is explicitly assigned weight 1.0.
    """

    EQUAL_FEATURE_WEIGHT = 1.0
    EQUAL_GROUP_WEIGHT = 1.0

    def __init__(
        self,
        rules: Mapping[
            str,
            Mapping[str, str],
        ] | None = None,
        *,
        top_n: int | None = CORE_LONG_TOP_N,
    ) -> None:
        if rules is None:
            rules = DEFAULT_LONG_SCORE_GROUPS

        # Equal weighting is explicit at both levels and does not depend
        # on LongGroupScore default weights.
        equal_rules = {
            group: {
                feature: {
                    "direction": direction,
                    "weight": self.EQUAL_FEATURE_WEIGHT,
                }
                for feature, direction
                in feature_directions.items()
            }
            for group, feature_directions
            in rules.items()
        }

        equal_group_weights = {
            group: self.EQUAL_GROUP_WEIGHT
            for group in rules
        }

        super().__init__(
            equal_rules,
            group_weights=equal_group_weights,
            top_n=top_n,
        )


def _normalize_horizons(
    horizons: Iterable[int],
) -> tuple[int, ...]:
    """Validate and normalize forward-return horizons in calendar months."""

    normalized: list[int] = []
    seen: set[int] = set()

    for horizon in horizons:
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon <= 0
        ):
            raise ValueError(
                "horizons must contain positive integer months"
            )

        if horizon not in seen:
            normalized.append(horizon)
            seen.add(horizon)

    if not normalized:
        raise ValueError(
            "at least one forward-return horizon is required"
        )

    return tuple(normalized)


def _normalize_ticker_label(value: object) -> str:
    """Normalize numeric ticker labels while leaving other symbols unchanged."""

    label = str(value)
    return label.zfill(6) if label.isdigit() else label


def _calculate_forward_returns(
    data: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizons: Iterable[int],
    ticker_col: str,
    as_of_col: str,
    price_date_col: str,
) -> pd.DataFrame:
    """Append forward returns to ticker-level data."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame"
        )

    if not isinstance(prices, pd.DataFrame):
        raise TypeError(
            "prices must be a pandas DataFrame"
        )

    if data.empty:
        raise ValueError(
            "data contains no rows to evaluate"
        )

    missing_data = {
        ticker_col,
        as_of_col,
    }.difference(data.columns)

    if missing_data:
        raise ValueError(
            "data is missing required columns: "
            f"{sorted(missing_data)}"
        )

    if price_date_col not in prices.columns:
        raise ValueError(
            f"prices must contain date column {price_date_col!r}"
        )

    as_of_values = pd.to_datetime(
        data[as_of_col],
        errors="raise",
    ).dt.normalize()

    if as_of_values.isna().any():
        raise ValueError(
            f"data contains missing {as_of_col!r} values"
        )

    unique_as_of = as_of_values.unique()

    if len(unique_as_of) != 1:
        raise ValueError(
            f"data must contain exactly one {as_of_col!r}; "
            f"found {len(unique_as_of)}"
        )

    as_of = pd.Timestamp(
        unique_as_of[0]
    )

    price_data = prices.copy()

    price_data[price_date_col] = pd.to_datetime(
        price_data[price_date_col],
        errors="raise",
    ).dt.normalize()

    price_data = (
        price_data
        .sort_values(price_date_col)
        .drop_duplicates(
            price_date_col,
            keep="last",
        )
    )

    rename_map = {
        column: _normalize_ticker_label(column)
        for column in price_data.columns
        if column != price_date_col
    }

    price_data = price_data.rename(
        columns=rename_map
    )

    future_dates = price_data.loc[
        price_data[price_date_col] > as_of,
        price_date_col,
    ]

    if future_dates.empty:
        raise ValueError(
            "price history contains no dates after as_of_date "
            f"{as_of.date().isoformat()}"
        )

    evaluated = data.copy()

    normalized_tickers = evaluated[
        ticker_col
    ].map(
        _normalize_ticker_label
    )

    evaluated["entry_date"] = pd.NaT
    evaluated["entry_price"] = np.nan

    for horizon in horizons:
        evaluated[
            f"exit_date_{horizon}m"
        ] = pd.NaT

        evaluated[
            f"return_{horizon}m"
        ] = np.nan

    max_price_date = price_data[
        price_date_col
    ].max()

    for row_index, ticker in normalized_tickers.items():
        if ticker not in price_data.columns:
            continue

        ticker_prices = (
            price_data[
                [
                    price_date_col,
                    ticker,
                ]
            ]
            .dropna(
                subset=[ticker]
            )
            .copy()
        )

        ticker_prices[ticker] = pd.to_numeric(
            ticker_prices[ticker],
            errors="coerce",
        )

        ticker_prices = ticker_prices.dropna(
            subset=[ticker]
        )

        entry_rows = ticker_prices.loc[
            ticker_prices[price_date_col] > as_of
        ]

        if entry_rows.empty:
            continue

        entry_row = entry_rows.iloc[0]

        entry_date = pd.Timestamp(
            entry_row[price_date_col]
        )

        entry_price = float(
            entry_row[ticker]
        )

        if (
            not np.isfinite(entry_price)
            or entry_price <= 0
        ):
            continue

        evaluated.at[
            row_index,
            "entry_date",
        ] = entry_date

        evaluated.at[
            row_index,
            "entry_price",
        ] = entry_price

        for horizon in horizons:
            target_date = (
                as_of
                + pd.DateOffset(
                    months=horizon
                )
            )

            if max_price_date < target_date:
                continue

            exit_rows = ticker_prices.loc[
                ticker_prices[price_date_col]
                >= target_date
            ]

            if exit_rows.empty:
                continue

            exit_row = exit_rows.iloc[0]

            exit_date = pd.Timestamp(
                exit_row[price_date_col]
            )

            exit_price = float(
                exit_row[ticker]
            )

            if (
                not np.isfinite(exit_price)
                or exit_price <= 0
            ):
                continue

            evaluated.at[
                row_index,
                f"exit_date_{horizon}m",
            ] = exit_date

            evaluated.at[
                row_index,
                f"return_{horizon}m",
            ] = (
                exit_price / entry_price
                - 1.0
            )

    return evaluated


def _compare_to_universe(
    universe_evaluated: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    horizons: Iterable[int],
    quantiles: int,
    ticker_col: str,
    as_of_col: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Evaluate selected stocks against an evaluated universe."""

    if not isinstance(selected, pd.DataFrame):
        raise TypeError(
            "selected must be a pandas DataFrame"
        )

    if selected.empty:
        raise ValueError(
            "selected contains no rows to compare"
        )

    if (
        isinstance(quantiles, bool)
        or not isinstance(quantiles, int)
        or quantiles < 2
    ):
        raise ValueError(
            "quantiles must be an integer "
            "greater than or equal to 2"
        )

    required = {
        ticker_col,
        as_of_col,
    }

    missing_selected = required.difference(
        selected.columns
    )

    if missing_selected:
        raise ValueError(
            "selected is missing required columns: "
            f"{sorted(missing_selected)}"
        )

    selected_as_of_values = pd.to_datetime(
        selected[as_of_col],
        errors="raise",
    ).dt.normalize()

    if selected_as_of_values.isna().any():
        raise ValueError(
            f"selected contains missing {as_of_col!r} values"
        )

    unique_selected_as_of = (
        selected_as_of_values.unique()
    )

    if len(unique_selected_as_of) != 1:
        raise ValueError(
            f"selected must contain exactly one {as_of_col!r}; "
            f"found {len(unique_selected_as_of)}"
        )

    selected_as_of = pd.Timestamp(
        unique_selected_as_of[0]
    )

    universe_as_of = pd.Timestamp(
        pd.to_datetime(
            universe_evaluated[as_of_col],
            errors="raise",
        )
        .dt.normalize()
        .iloc[0]
    )

    if selected_as_of != universe_as_of:
        raise ValueError(
            "selected and universe must use the same as_of_date; "
            f"selected={selected_as_of.date().isoformat()}, "
            f"universe={universe_as_of.date().isoformat()}"
        )

    selected_keys = selected[
        ticker_col
    ].map(
        _normalize_ticker_label
    )

    universe_keys = universe_evaluated[
        ticker_col
    ].map(
        _normalize_ticker_label
    )

    if selected_keys.duplicated().any():
        raise ValueError(
            "selected must contain at most one row per ticker"
        )

    if universe_keys.duplicated().any():
        raise ValueError(
            "universe must contain at most one row per ticker"
        )

    missing_tickers = sorted(
        set(selected_keys).difference(
            universe_keys
        )
    )

    if missing_tickers:
        suffix = (
            " ..."
            if len(missing_tickers) > 10
            else ""
        )

        raise ValueError(
            "selected contains tickers missing from universe: "
            f"{missing_tickers[:10]}{suffix}"
        )

    universe_evaluated = (
        universe_evaluated.copy()
    )

    universe_evaluated[
        "__ticker_key__"
    ] = universe_keys

    for horizon in horizons:
        return_col = (
            f"return_{horizon}m"
        )

        percentile_col = (
            f"return_{horizon}m_pct"
        )

        quantile_col = (
            f"return_{horizon}m_quantile"
        )

        returns = pd.to_numeric(
            universe_evaluated[
                return_col
            ],
            errors="coerce",
        )

        valid = returns.notna()

        universe_evaluated[
            percentile_col
        ] = np.nan

        universe_evaluated[
            quantile_col
        ] = pd.Series(
            pd.NA,
            index=universe_evaluated.index,
            dtype="string",
        )

        if valid.any():
            percentiles = (
                returns.loc[valid]
                .rank(
                    method="average",
                    pct=True,
                )
                * 100.0
            )

            universe_evaluated.loc[
                valid,
                percentile_col,
            ] = percentiles

            buckets = np.ceil(
                percentiles
                * quantiles
                / 100.0
            ).astype(
                int
            ).clip(
                1,
                quantiles,
            )

            universe_evaluated.loc[
                valid,
                quantile_col,
            ] = (
                "Q"
                + buckets.astype(str)
            )

    lookup = universe_evaluated.set_index(
        "__ticker_key__"
    )

    details = selected.copy()

    detail_keys = details[
        ticker_col
    ].map(
        _normalize_ticker_label
    )

    evaluation_columns = [
        "entry_date",
        "entry_price",
    ]

    for horizon in horizons:
        evaluation_columns.extend(
            [
                f"exit_date_{horizon}m",
                f"return_{horizon}m",
                f"return_{horizon}m_pct",
                f"return_{horizon}m_quantile",
            ]
        )

    for column in evaluation_columns:
        details[column] = (
            detail_keys.map(
                lookup[column]
            )
        )

    distribution_rows: list[
        dict[str, object]
    ] = []

    summary_rows: list[
        dict[str, object]
    ] = []

    baseline = 1.0 / quantiles

    for horizon in horizons:
        return_col = (
            f"return_{horizon}m"
        )

        percentile_col = (
            f"return_{horizon}m_pct"
        )

        quantile_col = (
            f"return_{horizon}m_quantile"
        )

        selected_returns = pd.to_numeric(
            details[return_col],
            errors="coerce",
        ).dropna()

        universe_returns = pd.to_numeric(
            universe_evaluated[
                return_col
            ],
            errors="coerce",
        ).dropna()

        valid_selected = (
            details[quantile_col]
            .notna()
        )

        valid_selected_count = int(
            valid_selected.sum()
        )

        for quantile in range(
            1,
            quantiles + 1,
        ):
            label = f"Q{quantile}"

            count = int(
                (
                    details.loc[
                        valid_selected,
                        quantile_col,
                    ]
                    == label
                ).sum()
            )

            distribution_rows.append(
                {
                    "horizon": f"{horizon}m",
                    "quantile": label,
                    "selected_count": count,
                    "selected_proportion": (
                        count
                        / valid_selected_count
                        if valid_selected_count > 0
                        else np.nan
                    ),
                    "baseline_proportion": baseline,
                }
            )

        selected_percentiles = pd.to_numeric(
            details[
                percentile_col
            ],
            errors="coerce",
        ).dropna()

        top_label = (
            f"Q{quantiles}"
        )

        bottom_label = "Q1"

        top_count = int(
            (
                details.loc[
                    valid_selected,
                    quantile_col,
                ]
                == top_label
            ).sum()
        )

        bottom_count = int(
            (
                details.loc[
                    valid_selected,
                    quantile_col,
                ]
                == bottom_label
            ).sum()
        )

        selected_mean = (
            selected_returns.mean()
            if not selected_returns.empty
            else np.nan
        )

        universe_mean = (
            universe_returns.mean()
            if not universe_returns.empty
            else np.nan
        )

        selected_median = (
            selected_returns.median()
            if not selected_returns.empty
            else np.nan
        )

        universe_median = (
            universe_returns.median()
            if not universe_returns.empty
            else np.nan
        )

        summary_rows.append(
            {
                "horizon": f"{horizon}m",
                "selected_count": int(
                    selected_returns.size
                ),
                "universe_count": int(
                    universe_returns.size
                ),
                "selected_mean_return": selected_mean,
                "universe_mean_return": universe_mean,
                "mean_excess_return": (
                    selected_mean
                    - universe_mean
                ),
                "selected_median_return": selected_median,
                "universe_median_return": universe_median,
                "median_excess_return": (
                    selected_median
                    - universe_median
                ),
                "top_quantile_hit_rate": (
                    top_count
                    / valid_selected_count
                    if valid_selected_count > 0
                    else np.nan
                ),
                "bottom_quantile_hit_rate": (
                    bottom_count
                    / valid_selected_count
                    if valid_selected_count > 0
                    else np.nan
                ),
                "quantile_baseline_rate": baseline,
                "selected_mean_percentile": (
                    selected_percentiles.mean()
                    if not selected_percentiles.empty
                    else np.nan
                ),
            }
        )

    distribution = pd.DataFrame(
        distribution_rows
    )

    summary = pd.DataFrame(
        summary_rows
    )

    return (
        details,
        distribution,
        summary,
    )


def _plot_forward_returns(
    evaluated: pd.DataFrame,
    *,
    horizons: Iterable[int],
    ticker_col: str,
    rank_col: str,
    ax: Any,
) -> Any:
    """Plot ticker-level forward returns on an existing axes."""

    from matplotlib.ticker import PercentFormatter

    return_columns = [
        f"return_{horizon}m"
        for horizon in horizons
    ]

    if (
        evaluated[
            return_columns
        ]
        .notna()
        .sum()
        .sum()
        == 0
    ):
        raise ValueError(
            "evaluated contains no calculated forward returns"
        )

    x = np.arange(
        len(horizons)
    )

    for _, row in evaluated.iterrows():
        ticker = str(
            row[ticker_col]
        )

        if (
            rank_col in evaluated.columns
            and pd.notna(
                row[rank_col]
            )
        ):
            rank_value = row[
                rank_col
            ]

            if (
                isinstance(
                    rank_value,
                    (int, np.integer),
                )
                or (
                    isinstance(
                        rank_value,
                        float,
                    )
                    and rank_value.is_integer()
                )
            ):
                rank_value = int(
                    rank_value
                )

            label = (
                f"#{rank_value} {ticker}"
            )

        else:
            label = ticker

        y = [
            row[column]
            for column in return_columns
        ]

        ax.plot(
            x,
            y,
            marker="o",
            label=label,
        )

    ax.axhline(
        0.0,
        linewidth=0.8,
    )

    ax.set_xticks(
        x,
        [
            f"{horizon}M"
            for horizon in horizons
        ],
    )

    ax.set_xlabel(
        "Forward horizon"
    )

    ax.set_ylabel(
        "Return"
    )

    ax.set_title(
        "Forward Returns by Selected Stock"
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )

    ax.legend(
        title="Selection rank"
    )

    return ax


def _plot_forward_return_distribution(
    distribution: pd.DataFrame,
    *,
    ax: Any,
) -> Any:
    """Plot selected-stock return quantiles on an existing axes."""

    data = distribution.copy()

    horizons = (
        data["horizon"]
        .drop_duplicates()
        .tolist()
    )

    quantiles = sorted(
        data[
            "quantile"
        ].unique(),
        key=lambda value: int(
            value.removeprefix("Q")
        ),
    )

    x = np.arange(
        len(quantiles)
    )

    width = (
        0.8
        / len(horizons)
    )

    for i, horizon in enumerate(
        horizons
    ):
        subset = (
            data.loc[
                data["horizon"]
                == horizon
            ]
            .set_index(
                "quantile"
            )
            .reindex(
                quantiles
            )
        )

        offset = (
            i
            - (
                len(horizons)
                - 1
            )
            / 2
        ) * width

        ax.bar(
            x + offset,
            subset[
                "selected_proportion"
            ]
            * 100,
            width=width,
            label=horizon,
        )

    baseline = (
        data[
            "baseline_proportion"
        ]
        .dropna()
        .iloc[0]
        * 100
    )

    ax.axhline(
        baseline,
        linestyle="--",
        linewidth=1,
        label=(
            "Universe baseline "
            f"({baseline:.0f}%)"
        ),
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        quantiles
    )

    ax.set_xlabel(
        "Universe forward-return quantile"
    )

    ax.set_ylabel(
        "Selected stocks (%)"
    )

    ax.set_title(
        "Selected Stocks by Forward-Return Quantile"
    )

    ax.legend()

    return ax


def evaluate_selection(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    universe: pd.DataFrame | None = None,
    horizons: Iterable[int] = (1, 3, 6),
    quantiles: int = 5,
    result: str = "detail",
    plot: str | bool = False,
    ticker_col: str = "ticker",
    as_of_col: str = "as_of_date",
    price_date_col: str = "date",
    rank_col: str = "selection_rank",
) -> Any:
    """Evaluate a selection with optional universe comparison and plotting.

    Parameters
    ----------
    result
        Used when ``plot=False``:
        - ``"detail"``: ticker-level forward-return details.
        - ``"distribution"``: selected stocks by universe return quantile.
        - ``"summary"``: selected-vs-universe summary statistics.
        - ``"all"``: details, distribution, and summary.

        ``"distribution"``, ``"summary"``, and ``"all"`` require
        ``universe``. If ``universe`` is supplied with ``result="detail"``,
        universe-relative percentile and quantile columns are also included.

    plot
        - ``False``: return the requested calculation result.
        - ``"returns"``: return one forward-return figure.
        - ``"distribution"``: return one quantile-distribution figure.
        - ``"both"``: return one figure containing both plots as subplots.

        Plot modes return figure objects rather than calculation results.
        ``"distribution"`` and ``"both"`` require ``universe``.
    """

    valid_results = {
        "detail",
        "distribution",
        "summary",
        "all",
    }

    if result not in valid_results:
        raise ValueError(
            "result must be one of "
            f"{sorted(valid_results)}"
        )

    valid_plots = {
        "returns",
        "distribution",
        "both",
    }

    if (
        plot is not False
        and plot not in valid_plots
    ):
        raise ValueError(
            "plot must be False, "
            "'returns', 'distribution', or 'both'"
        )

    horizons = _normalize_horizons(
        horizons
    )

    need_universe = (
        plot in {
            "distribution",
            "both",
        }
        or (
            plot is False
            and (
                universe is not None
                or result != "detail"
            )
        )
    )

    # Selected-only path.
    if not need_universe:
        evaluated = _calculate_forward_returns(
            selected,
            prices,
            horizons=horizons,
            ticker_col=ticker_col,
            as_of_col=as_of_col,
            price_date_col=price_date_col,
        )

        if plot == "returns":
            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            _plot_forward_returns(
                evaluated,
                horizons=horizons,
                ticker_col=ticker_col,
                rank_col=rank_col,
                ax=ax,
            )

            fig.tight_layout()

            return fig, ax

        return evaluated

    # Universe-relative path.
    if universe is None:
        raise ValueError(
            "universe is required for "
            "universe-relative evaluation"
        )

    universe_evaluated = (
        _calculate_forward_returns(
            universe,
            prices,
            horizons=horizons,
            ticker_col=ticker_col,
            as_of_col=as_of_col,
            price_date_col=price_date_col,
        )
    )

    (
        details,
        distribution,
        summary,
    ) = _compare_to_universe(
        universe_evaluated,
        selected,
        horizons=horizons,
        quantiles=quantiles,
        ticker_col=ticker_col,
        as_of_col=as_of_col,
    )

    if plot == "distribution":
        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        _plot_forward_return_distribution(
            distribution,
            ax=ax,
        )

        fig.tight_layout()

        return fig, ax

    if plot == "both":
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(16, 5),
        )

        _plot_forward_returns(
            details,
            horizons=horizons,
            ticker_col=ticker_col,
            rank_col=rank_col,
            ax=axes[0],
        )

        _plot_forward_return_distribution(
            distribution,
            ax=axes[1],
        )

        fig.tight_layout()

        return fig, axes

    if result == "detail":
        return details

    if result == "distribution":
        return distribution

    if result == "summary":
        return summary

    return (
        details,
        distribution,
        summary,
    )
