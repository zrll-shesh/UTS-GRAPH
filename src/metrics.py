"""Offline evaluation: ground truth from the test fold, Hit@N and NDCG@N."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .vsm import Interactions


@dataclass(frozen=True)
class GroundTruth:
    relevant: dict[int, set[int]]
    stats: dict[str, float]


def build_ground_truth(test_events: pd.DataFrame, inter: Interactions) -> GroundTruth:
    """Relevant items of a user = items they interact with in P2 that they had NOT seen in P1.

    Only users and items that exist in the training graph can be scored; the share of test
    activity that falls outside the training catalogue is reported as 'cold' in the stats.
    """
    user_index, item_index = inter.user_index, inter.item_index
    test_pairs = test_events[["visitorid", "itemid"]].drop_duplicates()
    warm_user = test_pairs["visitorid"].isin(inter.user_ids)
    warm_item = test_pairs["itemid"].isin(inter.item_ids)
    usable = test_pairs[warm_user & warm_item]

    relevant: dict[int, set[int]] = {}
    for visitor, item in zip(usable["visitorid"], usable["itemid"]):
        row, col = user_index[int(visitor)], item_index[int(item)]
        if inter.matrix[row, col] == 0:
            relevant.setdefault(row, set()).add(col)

    stats = {
        "test_visitors": test_events["visitorid"].nunique(),
        "test_visitors_in_training_%": 100 * warm_user.groupby(test_pairs["visitorid"]).any().mean(),
        "test_items": test_events["itemid"].nunique(),
        "test_items_in_training_catalogue_%": 100 * test_pairs["itemid"].drop_duplicates().isin(inter.item_ids).mean(),
        "visitors_with_new_relevant_items": len(relevant),
    }
    return GroundTruth(relevant, stats)


def eligible_users(inter: Interactions, truth: GroundTruth, min_items: int = 3, max_items: int = 100) -> np.ndarray:
    """Users that can be evaluated meaningfully: enough history, not a bot, at least one new relevant item."""
    degree = np.diff(inter.matrix.indptr)
    rows = np.array(sorted(truth.relevant), dtype=int)
    keep = (degree[rows] >= min_items) & (degree[rows] <= max_items)
    return rows[keep]


def hit_at_n(ranked: list[int], relevant: set[int], n: int) -> float:
    return float(any(item in relevant for item in ranked[:n]))


def recall_at_n(ranked: list[int], relevant: set[int], n: int) -> float:
    return len(set(ranked[:n]) & relevant) / len(relevant) if relevant else 0.0


def dcg(gains: list[float]) -> float:
    return float(sum(g / np.log2(pos + 2) for pos, g in enumerate(gains)))


def ndcg_at_n(ranked: list[int], relevant: set[int], n: int) -> float:
    """Binary-relevance NDCG: DCG of the list divided by the DCG of a perfect list."""
    gains = [1.0 if item in relevant else 0.0 for item in ranked[:n]]
    ideal = dcg([1.0] * min(n, len(relevant)))
    return dcg(gains) / ideal if ideal > 0 else 0.0


def evaluate(recommender, inter: Interactions, truth: GroundTruth, user_rows, n: int = 5) -> pd.DataFrame:
    """One row of metrics per user for the given recommender."""
    records = []
    for row in user_rows:
        recs = recommender.recommend(int(row), n)
        ranked = [inter.item_index[int(i)] for i in recs["itemId"]]
        relevant = truth.relevant[int(row)]
        records.append(
            {
                "visitorId": int(inter.user_ids[row]),
                f"hit@{n}": hit_at_n(ranked, relevant, n),
                f"recall@{n}": recall_at_n(ranked, relevant, n),
                f"ndcg@{n}": ndcg_at_n(ranked, relevant, n),
                "knn_share": float((recs["source"] == "knn").mean()) if len(recs) else 0.0,
            }
        )
    return pd.DataFrame(records)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap confidence interval of the mean."""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
