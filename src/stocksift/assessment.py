"""Neutral stock assessment from price and ratio history.

The module performs input-history consolidation and feature engineering. It
intentionally avoids feature aggregation, ranking into a final stock score,
and buy/sell/trim decisions.

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


def _resolve_files(
    *files: str | Path,
    data_root: str | Path | None = None,
) -> list[Path]:
    """Resolve one ``*`` pattern or multiple explicit CSV files."""
    if not files:
        raise ValueError("at least two files are required for consolidation")

    root = Path.cwd() if data_root is None else Path(data_root).expanduser().resolve()
    names = [str(file) for file in files]

    if any(char in name for name in names for char in ("?", "[", "]")):
        raise ValueError("only '*' is supported as a wildcard")

    if any("*" in name for name in names):
        if len(files) != 1:
            raise ValueError("a wildcard pattern cannot be mixed with explicit files")
        if Path(names[0]).expanduser().is_absolute():
            raise ValueError("wildcard patterns must be relative to data_root")
        paths = [
            path.resolve()
            for path in root.glob(names[0])
            if path.is_file() and path.suffix.lower() == ".csv"
        ]
        if not paths:
            raise FileNotFoundError(
                f"no CSV files match pattern {names[0]!r} under {root}"
            )
    else:
        paths = []
        for file in files:
            path = Path(file).expanduser()
            path = path if path.is_absolute() else root / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"input file not found: {path}")
            if path.suffix.lower() != ".csv":
                raise ValueError(f"input file must be CSV: {path.name}")
            paths.append(path)

    if len(set(paths)) != len(paths):
        raise ValueError("duplicate input files are not allowed")
    if len(paths) < 2:
        raise ValueError(
            f"at least two files are required for consolidation; found {len(paths)}"
        )
    return paths


def _consolidate_long(
    datasets: list[tuple[Path, pd.DataFrame]],
    *,
    ticker_col: str,
    date_col: str,
    value_cols: list[str],
    min_check_dates: int = 10,
    max_check_dates: int = 20,
    tolerance: float = 0.0,
) -> pd.DataFrame:
    """Consolidate normalized long histories after boundary consistency checks."""
    if min_check_dates < 1:
        raise ValueError("min_check_dates must be at least 1")
    if max_check_dates < min_check_dates:
        raise ValueError(
            "max_check_dates must be greater than or equal to min_check_dates"
        )
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if not value_cols:
        raise ValueError("at least one value column is required")

    prepared = []
    for path, frame in datasets:
        if frame.empty:
            raise ValueError(f"input file contains no usable data: {path.name}")
        data = frame.copy()
        data[date_col] = pd.to_datetime(data[date_col], errors="raise")
        data[ticker_col] = data[ticker_col].astype("string").str.zfill(6)
        data = (
            data.sort_values([ticker_col, date_col])
            .drop_duplicates([ticker_col, date_col], keep="last")
        )
        prepared.append(
            (path, data, data[date_col].min(), data[date_col].max())
        )

    max_dates = pd.Series([item[3] for item in prepared])
    duplicated = max_dates[max_dates.duplicated(keep=False)]
    if not duplicated.empty:
        date = duplicated.iloc[0]
        count = int((max_dates == date).sum())
        raise ValueError(
            f"multiple files have the same max date: {date.date()} ({count} files)"
        )

    prepared.sort(key=lambda item: item[3])
    previous_path, current, _, _ = prepared[0]
    keys = [ticker_col, date_col]

    for new_path, newer, new_start, _ in prepared[1:]:
        old_end = current[date_col].max()
        overlap_dates = (
            pd.Index(current[date_col].unique())
            .intersection(pd.Index(newer[date_col].unique()))
            .sort_values()
        )
        n_overlap = len(overlap_dates)
        if n_overlap < min_check_dates:
            raise ValueError(
                f"insufficient overlap between {previous_path.name} and "
                f"{new_path.name}: {n_overlap} common dates; at least "
                f"{min_check_dates} required "
                f"(old_end={old_end.date()}, new_start={new_start.date()})"
            )

        n_check = min(n_overlap, max_check_dates)
        check_dates = overlap_dates[-n_check:]
        left = current.loc[current[date_col].isin(check_dates), keys + value_cols]
        right = newer.loc[newer[date_col].isin(check_dates), keys + value_cols]
        compared = left.merge(right, on=keys, suffixes=("_old", "_new"))

        n_compared = n_failed = 0
        for column in value_cols:
            old = pd.to_numeric(compared[f"{column}_old"], errors="raise").to_numpy(float)
            new = pd.to_numeric(compared[f"{column}_new"], errors="raise").to_numpy(float)
            valid = np.isfinite(old) & np.isfinite(new)
            old, new = old[valid], new[valid]
            if not len(old):
                continue
            scale = np.maximum(np.abs(old), np.abs(new))
            rel_diff = np.divide(
                np.abs(new - old),
                scale,
                out=np.zeros_like(scale),
                where=scale != 0,
            )
            n_compared += len(rel_diff)
            n_failed += int((rel_diff > tolerance).sum())

        if not n_compared:
            raise ValueError(
                f"no comparable values in overlap between {previous_path.name} "
                f"and {new_path.name}"
            )

        if n_failed:
            print(
                f"WARNING: {previous_path.name} -> {new_path.name}: "
                f"old_end={old_end.date()}, new_start={new_start.date()}; "
                f"{n_failed}/{n_compared} values exceed tolerance "
                f"({n_failed / n_compared:.2%}). Older history discarded; "
                f"restart={new_start.date()}."
            )
            current = newer.copy()
        else:
            print(
                f"INFO: {previous_path.name} -> {new_path.name}: "
                f"old_end={old_end.date()}, new_start={new_start.date()}; "
                f"overlap={n_overlap} dates, checked={n_check}, "
                f"compared={n_compared} values."
            )
            current = (
                pd.concat([current, newer], ignore_index=True)
                .sort_values(keys)
                .drop_duplicates(keys, keep="last")
            )
        previous_path = new_path

    return current.sort_values(keys).reset_index(drop=True)


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
    RECOMMENDED_HISTORY_MONTHS = 13

    def __init__(
        self,
        *,
        price_csv: str | Path | None = None,
        ratio_csv: str | Path | None = None,
        ticker_name_csv: str | Path | None = None,
        data_root: str | Path | None = None,
    ) -> None:
        self.config = self._load_config()
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
        self.ticker_names = None
    
        if (price_csv is None) != (ratio_csv is None):
            raise ValueError(
                "price_csv and ratio_csv must be provided together"
            )
    
        if price_csv is not None and ratio_csv is not None:
            self.load_inputs(
                price_csv=price_csv,
                ratio_csv=ratio_csv,
                data_root=data_root,
            )
    
        if ticker_name_csv is not None:
            self.load_ticker_names(
                ticker_name_csv,
                data_root=data_root
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

    def load_inputs(
        self,
        *,
        price_csv: str | Path,
        ratio_csv: str | Path,
        data_root: str | Path | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and store the CSV inputs described by the YAML catalog."""

        data_root = (
            Path.cwd()
            if data_root is None
            else Path(data_root).expanduser().resolve()
        )
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


    @classmethod
    def consolidate_ratios(
        cls,
        *files: str | Path,
        data_root: str | Path | None = None,
        min_check_dates: int = 10,
        max_check_dates: int = 20,
        tolerance: float = 0.0,
        save: bool = False,
        output: str | Path | None = None,
    ) -> pd.DataFrame:
        """Consolidate ratio CSV histories, preferring later-dated files."""
        paths = _resolve_files(*files, data_root=data_root)
        assessment = cls()
        ticker_col = assessment.ratio_cols["ticker"]
        date_col = assessment.ratio_cols["date"]

        datasets = []
        first_columns = None
        for path in paths:
            frame = pd.read_csv(
                path,
                dtype={ticker_col: "string", date_col: "string"},
            )
            if not {ticker_col, date_col}.issubset(frame.columns):
                raise ValueError(
                    f"ratio file is missing ticker/date columns: {path.name}"
                )
            columns = list(frame.columns)
            if first_columns is None:
                first_columns = columns
            elif set(columns) != set(first_columns):
                raise ValueError(
                    "ratio files must have identical columns: "
                    f"{path.name} differs from the first file"
                )
            datasets.append((path, frame[first_columns]))

        value_cols = [
            column
            for column in first_columns
            if column not in {ticker_col, date_col}
        ]
        result = _consolidate_long(
            datasets,
            ticker_col=ticker_col,
            date_col=date_col,
            value_cols=value_cols,
            min_check_dates=min_check_dates,
            max_check_dates=max_check_dates,
            tolerance=tolerance,
        )[first_columns]

        if output is not None and not save:
            raise ValueError("output can be specified only when save=True")
        if save:
            if output is None:
                raise ValueError("output is required when save=True")
            root = Path.cwd() if data_root is None else Path(data_root).expanduser().resolve()
            output_path = Path(output).expanduser()
            output_path = output_path if output_path.is_absolute() else root / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        return result

    @classmethod
    def consolidate_prices(
        cls,
        *files: str | Path,
        data_root: str | Path | None = None,
        min_check_dates: int = 10,
        max_check_dates: int = 20,
        tolerance: float = 0.0,
        save: bool = False,
        output: str | Path | None = None,
    ) -> pd.DataFrame:
        """Consolidate wide price CSV histories through a shared long form."""
        paths = _resolve_files(*files, data_root=data_root)
        assessment = cls()
        date_col = assessment.price_cols["date"]
        ticker_col, value_col = "ticker", "price"
        datasets = []

        for path in paths:
            frame = pd.read_csv(path, dtype={date_col: "string"})
            if date_col not in frame.columns:
                raise ValueError(
                    f"price file is missing date column {date_col!r}: {path.name}"
                )
            frame = frame.rename(
                columns={
                    column: str(column).zfill(6)
                    for column in frame.columns
                    if column != date_col
                }
            )
            if frame.columns.duplicated().any():
                raise ValueError(
                    "price file has duplicate ticker columns after normalization: "
                    f"{path.name}"
                )
            frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
            frame = frame.sort_values(date_col).drop_duplicates(date_col, keep="last")
            long = frame.melt(
                id_vars=[date_col],
                var_name=ticker_col,
                value_name=value_col,
            ).dropna(subset=[value_col])
            datasets.append((path, long))

        long_result = _consolidate_long(
            datasets,
            ticker_col=ticker_col,
            date_col=date_col,
            value_cols=[value_col],
            min_check_dates=min_check_dates,
            max_check_dates=max_check_dates,
            tolerance=tolerance,
        )
        result = (
            long_result.pivot(index=date_col, columns=ticker_col, values=value_col)
            .sort_index()
            .sort_index(axis=1)
            .reset_index()
        )
        result.columns.name = None

        if output is not None and not save:
            raise ValueError("output can be specified only when save=True")
        if save:
            if output is None:
                raise ValueError("output is required when save=True")
            root = Path.cwd() if data_root is None else Path(data_root).expanduser().resolve()
            output_path = Path(output).expanduser()
            output_path = output_path if output_path.is_absolute() else root / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        return result

    
    def load_ticker_names(
        self,
        ticker_name_csv: str | Path,
        *,
        data_root: str | Path | None = None,
        ticker_col = '종목코드',
        name_col = '종목명'
    ):
        """Load and store ticker-name mappings."""

        data_root = (
            Path.cwd()
            if data_root is None
            else Path(data_root).expanduser().resolve()
        )
        ticker_name_csv = data_root / ticker_name_csv
    
        df = pd.read_csv(
            ticker_name_csv,
            encoding="euc-kr",
        )
    
        self.ticker_names = (
            df.set_index(ticker_col)[name_col]
        )
    
        return self.ticker_names
        

    def assess(
        self,
        as_of: Optional[
            str | pd.Timestamp
        ] = None,
        *,
        print_msg: bool = True,
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

        # Use only tickers that are active at the assessment date.
        available_price_dates = p.loc[
            p[price_date_col] <= as_of_ts,
            price_date_col,
        ]
        
        if available_price_dates.empty:
            raise ValueError(
                "prices contain no data on or before "
                f"{as_of_ts.date()}"
            )
        
        price_as_of_ts = available_price_dates.max()
        
        ratio_tickers = set(
            r.loc[
                r[ratio_date_col] == as_of_ts,
                ticker_col,
            ].dropna()
        )
        
        price_row = (
            p.loc[
                p[price_date_col] == price_as_of_ts
            ]
            .iloc[-1]
            .drop(labels=price_date_col)
        )
        
        price_tickers = set(
            price_row.index[
                price_row.notna()
            ]
        )
        
        common_tickers = sorted(
            ratio_tickers & price_tickers
        )
        
        if not common_tickers:
            raise ValueError(
                "prices and ratios have no common tickers "
                f"at as_of_date {as_of_ts.date()}"
            )

        if print_msg:
            data_start = max(
                p[price_date_col].min(),
                r[ratio_date_col].min(),
            )
            history_months = (
                (as_of_ts.year - data_start.year) * 12
                + as_of_ts.month
                - data_start.month
                - (as_of_ts.day < data_start.day)
            )
            history_months = max(0, history_months)

            if history_months < self.RECOMMENDED_HISTORY_MONTHS:
                print(
                    f"WARNING: Only {history_months} months of history "
                    "are available; "
                    f"{self.RECOMMENDED_HISTORY_MONTHS} months are "
                    "recommended for complete feature generation."
                )
            else:
                analysis_months = (
                    history_months
                    - self.RECOMMENDED_HISTORY_MONTHS
                )
                print(
                    f"INFO: {history_months} months of history available; "
                    f"approximately {analysis_months} months are usable "
                    "for historical recipe evaluation or trajectory "
                    "analysis."
                )

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
