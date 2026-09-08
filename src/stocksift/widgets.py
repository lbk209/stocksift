"""Notebook controls for long stock selection.

The widget layer only orchestrates existing recipe factories. Selection rules,
weights, filtering, and ranking remain owned by ``recipes`` and ``policy``.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .policy import SelectionPolicy
from .recipes import (
    LONG_FILTER_REGISTRY,
    LONG_STRATEGY_REGISTRY,
    apply_selection,
    build_long_selection,
)

CORE_LONG_TOP_N = 20

__all__ = [
    "LongSelectionWidget",
    "long_selection_widget",
]


class LongSelectionWidget:
    """Interactive Filter -> Strategy -> Top N controls.

    If ``features`` is provided, filter dropdown labels include the filter-only
    eligible count for the current feature table, for example
    ``Mild (83/200)``. Without ``features``, filter labels show only the preset
    names. Strategy labels never show counts because strategies rank the
    already-eligible universe rather than define a separate eligibility set.

    ``label`` provides the current recipe display name. ``info`` provides
    detailed display metadata. ``policies`` always reflects the current widget
    choices and can be passed directly to other selection/evaluation functions.
    ``select()`` additionally requires a bound feature table.
    """

    def __init__(
        self,
        features: pd.DataFrame | None = None,
        *,
        filter: str = "mild",
        strategy: str = "balanced",
        top_n: int = CORE_LONG_TOP_N,
        tickers: Iterable[str] | None = None,
    ) -> None:
        widgets, display = _notebook_tools()

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

        if (
            isinstance(top_n, bool)
            or not isinstance(top_n, int)
            or top_n <= 0
        ):
            raise ValueError("top_n must be a positive integer")

        if features is not None and not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame or None")

        self.features = features
        self.tickers = tickers
        self._display = display

        universe_count = _universe_count(features, tickers=tickers)
        filter_options = _filter_options(
            features,
            tickers=tickers,
            universe_count=universe_count,
        )
        strategy_options = [
            (spec["label"], key)
            for key, spec in LONG_STRATEGY_REGISTRY.items()
        ]

        control_style = {"description_width": "initial"}

        self.filter_selector = widgets.Dropdown(
            options=filter_options,
            value=filter,
            description="Filter",
            style=control_style,
            layout=widgets.Layout(width="230px"),
        )
        self.strategy_selector = widgets.Dropdown(
            options=strategy_options,
            value=strategy,
            description="Strategy",
            style=control_style,
            layout=widgets.Layout(width="230px"),
        )
        if universe_count is None:
            self.top_n_selector = widgets.IntText(
                value=top_n,
                step=1,
                description="Top N",
                style=control_style,
                layout=widgets.Layout(width="100px"),
            )
        else:
            self.top_n_selector = widgets.BoundedIntText(
                value=top_n,
                min=1,
                max=max(universe_count, top_n, 1),
                step=1,
                description="Top N",
                style=control_style,
                layout=widgets.Layout(width="100px"),
            )

        self.widget = widgets.HBox(
            [
                self.filter_selector,
                self.strategy_selector,
                self.top_n_selector,
            ],
            layout=widgets.Layout(
                width="600px",
                align_items="center",
                justify_content="space-between",
            ),
        )

    @property
    def filter(self) -> str:
        """Current filter registry key."""
        return str(self.filter_selector.value)

    @property
    def strategy(self) -> str:
        """Current strategy registry key."""
        return str(self.strategy_selector.value)

    @property
    def top_n(self) -> int:
        """Current Top-N breadth."""
        value = int(self.top_n_selector.value)
        if value <= 0:
            raise ValueError("top_n must be a positive integer")
        return value

    @property
    def info(self) -> dict[str, object]:
        """Return display metadata for the current selection recipe."""
        filter_spec = LONG_FILTER_REGISTRY[self.filter]
        strategy_spec = LONG_STRATEGY_REGISTRY[self.strategy]

        return {
            "label": (
                f"{filter_spec['label']} + "
                f"{strategy_spec['label']}"
            ),
            "filter": {
                "label": filter_spec["label"],
                "desc": filter_spec["desc"],
            },
            "strategy": {
                "label": strategy_spec["label"],
                "desc": strategy_spec["desc"],
            },
            "top_n": self.top_n,
        }

    @property
    def label(self) -> str:
        """Return the display label for the current selection recipe."""
        return str(self.info["label"])

    @property
    def policies(self) -> list[SelectionPolicy]:
        """Build policies from the current widget choices."""
        return build_long_selection(
            filter=self.filter,
            strategy=self.strategy,
            top_n=self.top_n,
        )

    def select(
        self,
        *,
        show_count: bool = False,
    ) -> pd.DataFrame:
        """Apply the current widget choices to the bound feature table.

        ``show_count`` is diagnostic only. The widget itself intentionally
        displays only the filter eligibility count.
        """
        if self.features is None:
            raise ValueError(
                "features is required to run selection"
            )

        return apply_selection(
            self.features,
            policies=self.policies,
            tickers=self.tickers,
            show_count=show_count,
        )

    def display(self) -> "LongSelectionWidget":
        """Display the controls in the current notebook and return self."""
        self._display(self.widget)
        return self


def long_selection_widget(
    features: pd.DataFrame | None = None,
    *,
    filter: str = "mild",
    strategy: str = "balanced",
    top_n: int = CORE_LONG_TOP_N,
    tickers: Iterable[str] | None = None,
    show: bool = True,
) -> LongSelectionWidget:
    """Create notebook controls for choosing a long-selection pipeline.

    ``features`` is optional when the widget is used only to configure policies.
    Eligible counts and ``select()`` are available only when features are bound.

    Examples
    --------
    controls = long_selection_widget(features)
    selected = controls.select()

    controls = long_selection_widget()
    label = controls.label
    info = controls.info
    policies = controls.policies
    """
    controls = LongSelectionWidget(
        features,
        filter=filter,
        strategy=strategy,
        top_n=top_n,
        tickers=tickers,
    )

    if show:
        controls.display()

    return controls


def _filter_options(
    features: pd.DataFrame | None,
    *,
    tickers: Iterable[str] | None,
    universe_count: int | None,
) -> list[tuple[str, str]]:
    """Return filter labels, with eligible counts when features are available."""
    if features is None:
        return [
            (spec["label"], key)
            for key, spec in LONG_FILTER_REGISTRY.items()
        ]

    assert universe_count is not None

    options: list[tuple[str, str]] = []

    for key, spec in LONG_FILTER_REGISTRY.items():
        policy = spec["factory"]()
        eligible_count = len(
            policy.select(
                features,
                tickers=tickers,
            )
        )
        label = f"{spec['label']} ({eligible_count}/{universe_count})"
        options.append((label, key))

    return options


def _universe_count(
    features: pd.DataFrame | None,
    *,
    tickers: Iterable[str] | None,
) -> int | None:
    """Return the available universe size, or None when it is unknown."""
    if features is None:
        if tickers is None:
            return None
        if isinstance(tickers, str):
            return 1
        return len(list(dict.fromkeys(tickers)))

    if tickers is None:
        return len(features)

    if isinstance(tickers, str):
        return 1

    return len(list(dict.fromkeys(tickers)))


def _notebook_tools():
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError(
            "long_selection_widget requires ipywidgets and IPython"
        ) from exc

    return widgets, display
