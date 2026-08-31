"""Neutral stock assessment from price and ratio history.

The module performs feature engineering only. It intentionally avoids feature
aggregation, ranking into a final stock score, and buy/sell/trim decisions.

Column mappings and generated-feature metadata live in ``assessment.yaml``.
Calculation rules remain in Python.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

ASSESSMENT_CONFIG_FILE = "assessment.yaml"


PRICE_STRUCTURAL_KEYS = ("date",)

RATIO_STRUCTURAL_KEYS = ("ticker", "date")

RATIO_CORE_FACTOR_KEYS = (
    "bps",
    "per",
    "pbr",
    "eps",
)

RATIO_OPTIONAL_FACTOR_KEYS = (
    "div",
    "dps",
)

RATIO_FACTOR_KEYS = (
    RATIO_CORE_FACTOR_KEYS
    + RATIO_OPTIONAL_FACTOR_KEYS
)

# All factor mappings supported by the YAML catalog.
RATIO_CATALOG_KEYS = (
    RATIO_STRUCTURAL_KEYS
    + RATIO_FACTOR_KEYS
)

# Columns that must actually exist in ratio input data.
RATIO_REQUIRED_KEYS = (
    RATIO_STRUCTURAL_KEYS
    + RATIO_CORE_FACTOR_KEYS
)


class StockAssessment:
    """Create neutral stock-level assessment features.

    The instance stores the YAML-backed column/feature catalog together with
    loaded price and ratio inputs. Results are generated on demand and are not
    kept as object state.
    """

    # Calculation rules intentionally remain in Python rather than YAML.
    VALUATION_HISTORY_MONTHS = 12
    VALUATION_CHANGE_MONTHS = 12
    VALUATION_SMOOTHING_DAYS = 20
    FUNDAMENTAL_LOOKBACK_MONTHS = 12
    MOMENTUM_SHORT_MONTHS = 6
    MOMENTUM_LONG_MONTHS = 12
    VOLATILITY_DAYS = 60

    def __init__(
        self,
        data_root: str | Path | None = None,
        *,
        price_csv: str | Path | None = None,
        ratio_csv: str | Path | None = None,
    ) -> None:
        self.config = self._load_config()

        self.set_data_root(data_root)

        self.price_cols = self._input_columns("prices")
        self.ratio_cols = self._input_columns("ratios")
        self.feature_groups = self._feature_groups()
        self.feature_cols = {
            key: key
            for group in self.feature_groups.values()
            for key in group
        }
        self.feature_labels = {
            key: meta["label"]
            for group in self.feature_groups.values()
            for key, meta in group.items()
        }

        self._validate_catalog()

        self.prices: pd.DataFrame | None = None
        self.ratios: pd.DataFrame | None = None

        if (price_csv is None) != (ratio_csv is None):
            raise ValueError(
                "price_csv and ratio_csv must be provided together"
            )

        if price_csv is not None and ratio_csv is not None:
            self.load_inputs(
                price_csv,
                ratio_csv,
            )

    @staticmethod
    def _load_config() -> dict:
        path = Path(__file__).with_name(ASSESSMENT_CONFIG_FILE)

        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError(
                "stock-assessment YAML must contain a mapping"
            )

        return config

    def _input_columns(
        self,
        source: str,
    ) -> dict[str, str]:
        try:
            entries = self.config[
                "input_columns"
            ][source]["columns"]
        except KeyError as exc:
            raise ValueError(
                f"missing input column configuration for {source!r}"
            ) from exc

        return {
            key: value["column"]
            for key, value in entries.items()
        }

    def _feature_groups(
        self,
    ) -> dict[str, dict[str, dict]]:
        features = self.config.get("features")

        if (
            not isinstance(features, dict)
            or not features
        ):
            raise ValueError(
                "YAML must define non-empty 'features' groups"
            )

        return features

    def _validate_catalog(self) -> None:
        """Validate the YAML catalog structure and input mappings."""

        missing_price = set(
            PRICE_STRUCTURAL_KEYS
        ).difference(
            self.price_cols
        )

        missing_ratio = set(
            RATIO_CATALOG_KEYS
        ).difference(
            self.ratio_cols
        )

        if missing_price:
            raise ValueError(
                "YAML is missing price semantic keys: "
                f"{sorted(missing_price)}"
            )

        if missing_ratio:
            raise ValueError(
                "YAML is missing ratio semantic keys: "
                f"{sorted(missing_ratio)}"
            )

        for group_name, features in self.feature_groups.items():
            for feature_key, meta in features.items():
                if not isinstance(meta, dict):
                    raise ValueError(
                        f"feature {group_name}.{feature_key} "
                        "must contain metadata"
                    )

                if not meta.get("label"):
                    raise ValueError(
                        f"feature {group_name}.{feature_key} "
                        "is missing 'label'"
                    )

                if not meta.get("description"):
                    raise ValueError(
                        f"feature {group_name}.{feature_key} "
                        "is missing 'description'"
                    )

    def set_data_root(
        self,
        data_root: str | Path | None = None,
    ) -> None:
        """Set the base directory used to resolve input data paths.

        If None, input paths are resolved from the current working directory
        when load_inputs() is called.
        """
        if data_root is None:
            self.data_root = None
            return

        path = Path(data_root).expanduser()

        if not path.is_absolute():
            path = Path.cwd() / path

        self.data_root = path.resolve()

    def load_inputs(
        self,
        price_csv: str | Path,
        ratio_csv: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and store the CSV inputs described by the YAML catalog."""
        data_root = self.data_root or Path.cwd()

        price_path = data_root / price_csv
        ratio_path = data_root / ratio_csv

        missing = []

        if not price_path.is_file():
            missing.append(
                f"price data: {price_path}"
            )

        if not ratio_path.is_file():
            missing.append(
                f"ratio data: {ratio_path}"
            )

        if missing:
            raise FileNotFoundError(
                "Input data not found:\n"
                + "\n".join(
                    f"  - {item}"
                    for item in missing
                )
                + f"\nData root: {data_root}"
            )

        price_date = self.price_cols["date"]
        ratio_ticker = self.ratio_cols["ticker"]
        ratio_date = self.ratio_cols["date"]

        self.prices = pd.read_csv(
            price_path,
            dtype={
                price_date: "string",
            },
        )

        self.ratios = pd.read_csv(
            ratio_path,
            dtype={
                ratio_ticker: "string",
                ratio_date: "string",
            },
        )

        return self.prices, self.ratios

    def assess(
        self,
        *,
        as_of: Optional[
            str | pd.Timestamp
        ] = None,
    ) -> pd.DataFrame:
        """Assess all tickers shared by the loaded price and ratio inputs.

        Returns one row per ticker. No feature-family composite score and no
        trading decision are produced.
        """
        if self.prices is None or self.ratios is None:
            raise ValueError(
                "inputs are not loaded; call load_inputs() first"
            )

        p, r = self._prepare_inputs(
            self.prices,
            self.ratios,
        )

        pc = self.price_cols
        rc = self.ratio_cols
        f = self.feature_cols

        price_date_col = pc["date"]
        ratio_date_col = rc["date"]
        ticker_col = rc["ticker"]

        common_tickers = sorted(
            (
                set(p.columns)
                - {price_date_col}
            )
            & set(
                r[ticker_col].dropna()
            )
        )

        if not common_tickers:
            raise ValueError(
                "prices and ratios have no common tickers"
            )

        latest_common_date = min(
            p[price_date_col].max(),
            r[ratio_date_col].max(),
        )

        requested_as_of = (
            pd.Timestamp(as_of)
            if as_of is not None
            else latest_common_date
        )

        requested_as_of = min(
            requested_as_of,
            latest_common_date,
        )

        available_ratio_dates = r.loc[
            r[ratio_date_col]
            <= requested_as_of,
            ratio_date_col,
        ]

        if available_ratio_dates.empty:
            raise ValueError(
                "ratios contain no data on or before "
                f"{requested_as_of.date()}"
            )

        as_of_ts = available_ratio_dates.max()

        current = self._ratio_snapshot(
            r,
            as_of_ts,
        ).reindex(
            common_tickers
        )

        fundamental_base_date = (
            as_of_ts
            - pd.DateOffset(
                months=self.FUNDAMENTAL_LOOKBACK_MONTHS
            )
        )

        previous = self._ratio_snapshot(
            r,
            fundamental_base_date,
        ).reindex(
            common_tickers
        )

        out = pd.DataFrame(
            index=pd.Index(
                common_tickers,
                name="ticker",
            )
        )

        out["as_of_date"] = (
            as_of_ts.date().isoformat()
        )

        out["price"] = [
            self._price_at_or_before(
                p,
                ticker,
                as_of_ts,
            )
            for ticker in common_tickers
        ]

        # Keep available point-in-time ratio/fundamental snapshots visible
        # under their configured input column names.
        for semantic_key in RATIO_FACTOR_KEYS:
            column = rc[semantic_key]

            if column in current.columns:
                out[column] = current[column]

        # ---- Valuation ---------------------------------------------------
        history_start = (
            as_of_ts
            - pd.DateOffset(
                months=self.VALUATION_HISTORY_MONTHS
            )
        )

        change_base_date = (
            as_of_ts
            - pd.DateOffset(
                months=self.VALUATION_CHANGE_MONTHS
            )
        )

        per_smoothed = {}
        pbr_smoothed = {}
        per_hist_pct = {}
        pbr_hist_pct = {}
        per_change = {}
        pbr_change = {}

        for ticker in common_tickers:
            per_s = self._smoothed_positive_ratio(
                r,
                ticker,
                rc["per"],
                as_of=as_of_ts,
            )

            pbr_s = self._smoothed_positive_ratio(
                r,
                ticker,
                rc["pbr"],
                as_of=as_of_ts,
            )

            current_per = (
                self._value_at_or_before(
                    per_s,
                    as_of_ts,
                )
            )

            current_pbr = (
                self._value_at_or_before(
                    pbr_s,
                    as_of_ts,
                )
            )

            base_per = (
                self._value_at_or_before(
                    per_s,
                    change_base_date,
                )
            )

            base_pbr = (
                self._value_at_or_before(
                    pbr_s,
                    change_base_date,
                )
            )

            per_smoothed[ticker] = current_per
            pbr_smoothed[ticker] = current_pbr

            per_hist_pct[ticker] = (
                self._own_history_percentile(
                    per_s,
                    start=history_start,
                    end=as_of_ts,
                )
            )

            pbr_hist_pct[ticker] = (
                self._own_history_percentile(
                    pbr_s,
                    start=history_start,
                    end=as_of_ts,
                )
            )

            per_change[ticker] = (
                self._relative_change(
                    current_per,
                    base_per,
                )
            )

            pbr_change[ticker] = (
                self._relative_change(
                    current_pbr,
                    base_pbr,
                )
            )

        out[
            f["per_smoothed"]
        ] = pd.Series(
            per_smoothed
        )

        out[
            f["pbr_smoothed"]
        ] = pd.Series(
            pbr_smoothed
        )

        out[
            f["per_hist_pct"]
        ] = pd.Series(
            per_hist_pct
        )

        out[
            f["pbr_hist_pct"]
        ] = pd.Series(
            pbr_hist_pct
        )

        out[
            f["per_change_12m"]
        ] = pd.Series(
            per_change
        )

        out[
            f["pbr_change_12m"]
        ] = pd.Series(
            pbr_change
        )

        out[
            f["per_expansion_pct"]
        ] = self._pct_rank(
            out[
                f["per_change_12m"]
            ]
        )

        out[
            f["pbr_expansion_pct"]
        ] = self._pct_rank(
            out[
                f["pbr_change_12m"]
            ]
        )

        # ---- Fundamentals ------------------------------------------------
        eps_col = rc["eps"]
        bps_col = rc["bps"]

        out[
            f["eps_growth_12m"]
        ] = self._safe_growth(
            out[eps_col],
            previous[eps_col],
        )

        out[
            f["bps_growth_12m"]
        ] = self._safe_growth(
            out[bps_col],
            previous[bps_col],
        )

        out[
            f["eps_growth_pct"]
        ] = self._pct_rank(
            out[
                f["eps_growth_12m"]
            ]
        )

        out[
            f["bps_growth_pct"]
        ] = self._pct_rank(
            out[
                f["bps_growth_12m"]
            ]
        )

        dps_col = rc["dps"]

        if dps_col in r.columns:
            out[
                f["dps_growth_12m"]
            ] = self._safe_growth(
                out[dps_col],
                previous[dps_col],
            )

            out[
                f["dps_growth_pct"]
            ] = self._pct_rank(
                out[
                    f["dps_growth_12m"]
                ]
            )

        # ---- Momentum ----------------------------------------------------
        short_date = (
            as_of_ts
            - pd.DateOffset(
                months=self.MOMENTUM_SHORT_MONTHS
            )
        )

        long_date = (
            as_of_ts
            - pd.DateOffset(
                months=self.MOMENTUM_LONG_MONTHS
            )
        )

        short_base = pd.Series(
            {
                ticker: self._price_at_or_before(
                    p,
                    ticker,
                    short_date,
                )
                for ticker in common_tickers
            }
        )

        long_base = pd.Series(
            {
                ticker: self._price_at_or_before(
                    p,
                    ticker,
                    long_date,
                )
                for ticker in common_tickers
            }
        )

        short_base.index.name = "ticker"
        long_base.index.name = "ticker"

        out[
            f["momentum_6m"]
        ] = (
            out["price"]
            / short_base
            - 1.0
        )

        out[
            f["momentum_12m"]
        ] = (
            out["price"]
            / long_base
            - 1.0
        )

        out[
            f["momentum_6m_pct"]
        ] = self._pct_rank(
            out[
                f["momentum_6m"]
            ]
        )

        out[
            f["momentum_12m_pct"]
        ] = self._pct_rank(
            out[
                f["momentum_12m"]
            ]
        )

        # ---- Stock risk --------------------------------------------------
        volatility = {}
        max_drawdown = {}

        for ticker in common_tickers:
            series = (
                p.loc[
                    p[price_date_col]
                    <= as_of_ts,
                    [
                        price_date_col,
                        ticker,
                    ],
                ]
                .dropna(
                    subset=[ticker]
                )
                .set_index(
                    price_date_col
                )[ticker]
                .astype(float)
            )

            returns = (
                series
                .pct_change()
                .dropna()
            )

            recent_returns = (
                returns.tail(
                    self.VOLATILITY_DAYS
                )
            )

            volatility[ticker] = (
                recent_returns.std(ddof=1)
                * np.sqrt(252)
                if len(recent_returns) >= 20
                else np.nan
            )

            one_year = series.loc[
                series.index
                >= long_date
            ]

            if len(one_year) >= 2:
                drawdowns = (
                    one_year
                    / one_year.cummax()
                    - 1.0
                )

                max_drawdown[ticker] = float(
                    drawdowns.min()
                )
            else:
                max_drawdown[ticker] = np.nan

        out[
            f["volatility_60d"]
        ] = pd.Series(
            volatility
        )

        out[
            f["max_drawdown_12m"]
        ] = pd.Series(
            max_drawdown
        )

        out[
            f["volatility_risk_pct"]
        ] = self._pct_rank(
            out[
                f["volatility_60d"]
            ]
        )

        out[
            f["drawdown_risk_pct"]
        ] = self._pct_rank(
            -out[
                f["max_drawdown_12m"]
            ]
        )

        return out.reset_index()

    def _prepare_inputs(
        self,
        prices: pd.DataFrame,
        ratios: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:
        pc = self.price_cols
        rc = self.ratio_cols

        price_date_col = pc["date"]

        required_ratio_columns = {
            rc[key]
            for key in RATIO_REQUIRED_KEYS
        }

        if price_date_col not in prices.columns:
            raise ValueError(
                "prices must contain configured "
                f"date column {price_date_col!r}"
            )

        missing = (
            required_ratio_columns
            .difference(
                ratios.columns
            )
        )

        if missing:
            raise ValueError(
                "ratios is missing configured columns: "
                f"{sorted(missing)}"
            )

        p = prices.copy()
        r = ratios.copy()

        p[
            price_date_col
        ] = pd.to_datetime(
            p[price_date_col],
            errors="raise",
        )

        r[
            rc["date"]
        ] = pd.to_datetime(
            r[rc["date"]],
            errors="raise",
        )

        r[
            rc["ticker"]
        ] = (
            r[rc["ticker"]]
            .astype("string")
            .str.zfill(6)
        )

        # All price columns except the configured date column are ticker IDs.
        p = p.rename(
            columns={
                column: str(
                    column
                ).zfill(6)
                for column in p.columns
                if column
                != price_date_col
            }
        )

        p = (
            p.sort_values(
                price_date_col
            )
            .drop_duplicates(
                price_date_col,
                keep="last",
            )
        )

        r = (
            r.sort_values(
                [
                    rc["ticker"],
                    rc["date"],
                ]
            )
            .drop_duplicates(
                [
                    rc["ticker"],
                    rc["date"],
                ],
                keep="last",
            )
        )

        return p, r

    def _ratio_snapshot(
        self,
        ratios: pd.DataFrame,
        target: pd.Timestamp,
    ) -> pd.DataFrame:
        rc = self.ratio_cols

        subset = ratios.loc[
            ratios[rc["date"]]
            <= target
        ]

        if subset.empty:
            return pd.DataFrame(
                columns=ratios.columns
            )

        return (
            subset
            .groupby(
                rc["ticker"],
                sort=False,
                as_index=False,
            )
            .tail(1)
            .set_index(
                rc["ticker"]
            )
        )

    def _price_at_or_before(
        self,
        prices: pd.DataFrame,
        ticker: str,
        target: pd.Timestamp,
    ) -> float:
        date_col = (
            self.price_cols["date"]
        )

        subset = prices.loc[
            prices[date_col]
            <= target,
            [
                date_col,
                ticker,
            ],
        ].dropna(
            subset=[ticker]
        )

        if subset.empty:
            return np.nan

        return float(
            subset.iloc[-1][ticker]
        )

    def _smoothed_positive_ratio(
        self,
        ratios: pd.DataFrame,
        ticker: str,
        column: str,
        *,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        rc = self.ratio_cols

        series = (
            ratios.loc[
                (
                    ratios[
                        rc["ticker"]
                    ]
                    == ticker
                )
                & (
                    ratios[
                        rc["date"]
                    ]
                    <= as_of
                ),
                [
                    rc["date"],
                    column,
                ],
            ]
            .dropna(
                subset=[column]
            )
            .set_index(
                rc["date"]
            )[column]
            .astype(float)
        )

        series = series.where(
            series > 0
        )

        return series.rolling(
            window=(
                self.VALUATION_SMOOTHING_DAYS
            ),
            min_periods=(
                self.VALUATION_SMOOTHING_DAYS
            ),
        ).median()

    @staticmethod
    def _safe_growth(
        current: pd.Series,
        previous: pd.Series,
    ) -> pd.Series:
        growth = (
            current
            / previous
            - 1.0
        )

        return growth.where(
            previous > 0
        )

    @staticmethod
    def _pct_rank(
        series: pd.Series,
    ) -> pd.Series:
        return (
            series.rank(
                method="average",
                pct=True,
            )
            * 100.0
        )

    @staticmethod
    def _value_at_or_before(
        series: pd.Series,
        target: pd.Timestamp,
    ) -> float:
        subset = (
            series.loc[
                series.index
                <= target
            ]
            .dropna()
        )

        if subset.empty:
            return np.nan

        return float(
            subset.iloc[-1]
        )

    @staticmethod
    def _own_history_percentile(
        series: pd.Series,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> float:
        window = (
            series.loc[
                (
                    series.index
                    >= start
                )
                & (
                    series.index
                    <= end
                )
            ]
            .dropna()
        )

        if window.empty:
            return np.nan

        return float(
            window.rank(
                method="average",
                pct=True,
            ).iloc[-1]
            * 100.0
        )

    @staticmethod
    def _relative_change(
        current: float,
        base: float,
    ) -> float:
        if (
            np.isfinite(current)
            and np.isfinite(base)
            and base > 0
        ):
            return (
                current
                / base
                - 1.0
            )

        return np.nan
