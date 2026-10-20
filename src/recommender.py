"""k-nearest-neighbour recommenders on top of the vector space models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .vsm import Interactions, VSM, build_vsm, log_tf, top_k_neighbours


class PopularityRecommender:
    """Baseline: recommend the items with the most distinct visitors that the user has not seen."""

    name = "Popularity"

    def __init__(self, inter: Interactions):
        self.inter = inter
        self.tf = log_tf(inter)
        self.order = np.argsort(-np.diff(self.tf.tocsc().indptr), kind="stable")

    def recommend(self, user_row: int, n: int = 5) -> pd.DataFrame:
        seen = set(self.tf[user_row].indices.tolist())
        picks = [col for col in self.order if col not in seen][:n]
        return _frame(self.inter, picks, [0.0] * len(picks), [0] * len(picks), "popularity")


class KNNRecommender:
    """User-based or item-based KNN, depending on the projection (`space`).

    user space: find the k most similar visitors, then score every item they touched by
                sum(similarity * their interaction strength).
    item space: for each item the visitor already touched (the "seeds", strongest first),
                find its k most similar items and accumulate similarity * seed strength.
    Items the visitor has already seen are never recommended.
    """

    def __init__(self, inter: Interactions, space: str, k: int = 5, min_support: int = 3, tfidf: bool = True, max_seeds: int = 20):
        self.inter, self.space, self.k = inter, space, k
        self.min_support, self.max_seeds = min_support, max_seeds
        self.tf = log_tf(inter)
        self.vsm: VSM = build_vsm(inter, space, tfidf)
        self.fallback = PopularityRecommender(inter)
        self.name = f"{space.capitalize()}-KNN (k={k})"
        self.last_neighbours = pd.DataFrame()

    def _score_user_based(self, user_row: int) -> tuple[np.ndarray, np.ndarray]:
        rows, sims = top_k_neighbours(self.vsm, [user_row], self.k, self.min_support)[user_row]
        self.last_neighbours = pd.DataFrame(
            {"visitorId": self.inter.user_ids[rows], "cosine": sims, "items_touched": self.vsm.support[rows]}
        )
        block = self.tf[rows]
        scores = np.asarray(block.T @ sims).ravel()
        supporters = np.asarray((block > 0).sum(axis=0)).ravel()
        return scores, supporters

    def _score_item_based(self, user_row: int) -> tuple[np.ndarray, np.ndarray]:
        profile = self.tf[user_row]
        strongest = np.argsort(-profile.data, kind="stable")[: self.max_seeds]
        seeds, strength = profile.indices[strongest], profile.data[strongest]
        neighbours = top_k_neighbours(self.vsm, seeds, self.k, self.min_support)
        scores = np.zeros(self.tf.shape[1])
        supporters = np.zeros(self.tf.shape[1], dtype=int)
        links = []
        for seed, weight in zip(seeds, strength):
            cols, sims = neighbours[int(seed)]
            scores[cols] += weight * sims
            supporters[cols] += 1
            links += [(self.inter.item_ids[seed], self.inter.item_ids[c], s) for c, s in zip(cols, sims)]
        self.last_neighbours = pd.DataFrame(links, columns=["seedItemId", "neighbourItemId", "cosine"])
        return scores, supporters

    def recommend(self, user_row: int, n: int = 5) -> pd.DataFrame:
        score_fn = self._score_user_based if self.space == "user" else self._score_item_based
        scores, supporters = score_fn(user_row)
        seen = self.tf[user_row].indices
        scores[seen] = 0.0
        candidates = np.flatnonzero(scores > 0)
        top = candidates[np.lexsort((candidates, -scores[candidates]))][:n]
        picks, values, support = top.tolist(), scores[top].tolist(), supporters[top].tolist()
        source = ["knn"] * len(picks)
        if len(picks) < n:
            filler = self.fallback.recommend(user_row, n + len(picks))
            for item_id in filler["itemId"]:
                col = self.inter.item_index[int(item_id)]
                if col not in picks and len(picks) < n:
                    picks.append(col), values.append(0.0), support.append(0), source.append("popularity")
        return _frame(self.inter, picks, values, support, source)


def _frame(inter: Interactions, cols, scores, supporters, source) -> pd.DataFrame:
    if isinstance(source, str):
        source = [source] * len(cols)
    frame = pd.DataFrame(
        {"rank": np.arange(1, len(cols) + 1), "itemId": inter.item_ids[np.asarray(cols, dtype=int)],
         "score": scores, "supporters": supporters, "source": source}
    )
    return frame


def similarity_pairs(vsm: VSM, rows: np.ndarray, k: int, min_support: int) -> pd.DataFrame:
    """Top-k neighbours for many rows, in the (source, target, score, rank) shape that Neo4j expects."""
    found = top_k_neighbours(vsm, rows, k, min_support)
    records = [
        {"source": int(vsm.ids[row]), "target": int(vsm.ids[col]), "score": float(s), "rank": rank}
        for row, (cols, sims) in found.items()
        for rank, (col, s) in enumerate(zip(cols, sims), start=1)
    ]
    return pd.DataFrame(records, columns=["source", "target", "score", "rank"])
