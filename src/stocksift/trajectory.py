"""High-level workflow for reviewing stock-selection trajectories.

TrajectoryExplorer is a thin facade over the existing selection, widget,
price-download, and chart APIs. The underlying functions remain independently
usable.
"""

from __future__ import annotations

from typing import Any, Sequence


__all__ = ["TrajectoryExplorer"]


class TrajectoryExplorer:
    """Coordinate trajectory selection and price review for one assessment.

    Parameters
    ----------
    assessment
        A prepared StockAssessment instance.

    Notes
    -----
    Heavy notebook/chart/data modules are imported only when their respective
    methods are called, keeping ``import stocksift`` relatively lightweight.
    """

    def __init__(self, assessment: Any) -> None:
        self.assessment = assessment
        self.controls = None
        self.selected = None
        self.tickers: list[str] = []
        self.prices = None

    def widget(self, features=None, **kwargs):
        """Create and store the long-selection controls.

        ``features`` may be None when the widget is used only to configure
        policies. Additional arguments are passed to ``long_selection_widget``.
        """
        from .widgets import long_selection_widget

        self.controls = long_selection_widget(features, **kwargs)
        return self.controls

    def run(
        self,
        *,
        as_of=None,
        policies: Sequence[Any] | None = None,
        top_n: int | None = None,
        freq: int = 5,
        periods: int = 5,
        style: bool = True,
        show_controls: bool = True,
    ):
        """Calculate and store the selection trajectory."""
        from .recipes import selection_trajectory
    
        if policies is None and self.controls is not None:
            policies = self.controls.policies
    
            if show_controls:
                info = self.controls.info
                print(
                    f"Controls: {info['label']} | "
                    f"Top N={info['top_n']}"
                )
    
        selected = selection_trajectory(
            self.assessment,
            as_of=as_of,
            policies=policies,
            top_n=top_n,
            freq=freq,
            periods=periods,
            style=style,
        )
    
        data = selected.data if hasattr(selected, "data") else selected
    
        if data.empty or data.shape[1] == 0:
            raise ValueError("trajectory result contains no selections")
    
        self.selected = selected
        self.tickers = (
            data.iloc[:, -1]
            .dropna()
            .astype(str)
            .tolist()
        )
    
        self.prices = None
    
        return selected



    def diagnose(self, *, as_of=None, policies=None):
        """Show historical coverage and threshold-filter diagnostics."""
        from .recipes import default_long_selection
    
        if policies is None:
            policies = (
                self.controls.policies
                if self.controls is not None
                else default_long_selection()
            )
    
        features = self.assessment.assess(
            as_of=as_of,
            print_msg=True,
        )
    
        policy = policies[0]
        if not hasattr(policy, "trace"):
            raise ValueError(
                "The first selection policy does not support trace()."
            )
    
        return policy.trace(features)

        

    def fetch_prices(
        self,
        *,
        start,
        end=None,
        errors: str = "ignore",
    ):
        """Download and store OHLCV data for the latest selected tickers."""
        from .naver import fetch_naver_ohlcv

        if not self.tickers:
            raise ValueError("run() must produce tickers before fetching prices")

        self.prices = fetch_naver_ohlcv(
            self.tickers,
            start=start,
            end=end,
            errors=errors,
        )
        return self.prices

    def browse(
        self,
        *,
        start=None,
        end=None,
        errors: str = "ignore",
        freq: str = "D",
        comparison: bool = False,
        chart_kwargs: dict[str, Any] | None = None,
        comparison_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Browse selected tickers with technical and comparison charts.

        If prices have not yet been loaded, ``start`` is required and OHLCV
        data are downloaded automatically. ``chart_kwargs`` and
        ``comparison_kwargs`` are passed to the corresponding chart browsers.
        """
        from .charts import browse_price_chart, browse_price_comparison

        if not self.tickers:
            raise ValueError("run() must produce tickers before browse()")

        if self.prices is None:
            if start is None:
                raise ValueError(
                    "start is required when prices have not been fetched"
                )
            self.fetch_prices(
                start=start,
                end=end,
                errors=errors,
            )

        ticker_names = getattr(
            self.assessment,
            "ticker_names",
            None,
        )

        chart_options = {
            "ticker_names": ticker_names,
            "freq": freq,
        }
        if chart_kwargs:
            chart_options.update(chart_kwargs)

        browse_price_chart(
            self.prices,
            self.tickers,
            **chart_options,
        )

        if comparison:
            comparison_options = {
                "ticker_names": ticker_names,
            }
            if comparison_kwargs:
                comparison_options.update(comparison_kwargs)

            browse_price_comparison(
                self.prices,
                self.tickers,
                **comparison_options,
            )
