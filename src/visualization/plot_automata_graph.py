"""
Probabilistic automata state graph visualization.

Only observed training transitions are drawn. Smoothed but unobserved
transitions are intentionally excluded from the graph to preserve
interpretability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx

from src.automata.probabilistic_automata import ProbabilisticAutomata


def plot_automata_graph(
    automata: ProbabilisticAutomata,
    max_edges: int = 30,
    minimum_probability: float = 0.0,
    show_edge_labels: bool = False,
    title: str = "Probabilistic Automata State Graph",
    save_path: Optional[str | Path] = None,
    dpi: int = 300
):
    """
    Plot observed automata transitions using NetworkX.

    Parameters
    ----------
    automata:
        Fitted probabilistic automata model.

    max_edges:
        Maximum number of observed transitions shown, ordered by
        transition probability.

    minimum_probability:
        Only observed transitions with probability greater than or equal
        to this value are displayed.

    show_edge_labels:
        Whether transition probabilities should be printed on graph edges.
    """
    if not automata.is_fitted_:
        raise RuntimeError("ProbabilisticAutomata must be fitted first.")

    if max_edges <= 0:
        raise ValueError("max_edges must be greater than zero.")

    if minimum_probability < 0 or minimum_probability > 1:
        raise ValueError(
            "minimum_probability must be between 0 and 1."
        )

    edge_records = []

    for from_state, outgoing_transitions in automata.transition_counts_.items():
        for to_state, count in outgoing_transitions.items():
            probability = automata.transition_probability(
                from_state=from_state,
                to_state=to_state
            )

            if probability >= minimum_probability:
                edge_records.append(
                    (from_state, to_state, probability, int(count))
                )

    edge_records.sort(
        key=lambda record: (record[2], record[3]),
        reverse=True
    )

    selected_edges = edge_records[:max_edges]

    graph = nx.DiGraph()
    graph.add_nodes_from(automata.states_)

    for from_state, to_state, probability, count in selected_edges:
        graph.add_edge(
            from_state,
            to_state,
            probability=probability,
            count=count
        )

    figure, axis = plt.subplots(figsize=(10, 8))

    positions = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=850,
        ax=axis
    )

    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=8,
        ax=axis
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        arrows=True,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        ax=axis
    )

    if show_edge_labels:
        edge_labels = {
            (from_state, to_state): f"{attributes['probability']:.3f}"
            for from_state, to_state, attributes in graph.edges(data=True)
        }

        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels=edge_labels,
            font_size=7,
            ax=axis
        )

    axis.set_title(title)
    axis.axis("off")

    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return figure, axis