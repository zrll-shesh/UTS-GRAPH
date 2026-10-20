"""Vector space models of the user-item graph, plus cosine nearest-neighbour search."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class Interactions:
    """Weighted user-item matrix (rows = visitors, columns = items) with its id lookups."""

    matrix: sp.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def user_index(self) -> dict[int, int]:
        return {int(u): row for row, u in enumerate(self.user_ids)}

    @property
    def item_index(self) -> dict[int, int]:
        return {int(i): col for col, i in enumerate(self.item_ids)}

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape


def build_interactions(edges: pd.DataFrame, rel_weights: dict[str, float]) -> Interactions:
    """Collapse the aggregated edge list (visitorId, itemId, rel, n) into one weighted cell per pair."""
    weight = edges["n"].to_numpy(float) * edges["rel"].map(rel_weights).to_numpy(float)
    pairs = edges.assign(weight=weight).groupby(["visitorId", "itemId"], as_index=False)["weight"].sum()
    user_ids, rows = np.unique(pairs["visitorId"].to_numpy(), return_inverse=True)
    item_ids, cols = np.unique(pairs["itemId"].to_numpy(), return_inverse=True)
    matrix = sp.csr_matrix((pairs["weight"].to_numpy(float), (rows, cols)), shape=(len(user_ids), len(item_ids)))
    return Interactions(matrix, user_ids, item_ids)


def binary(matrix: sp.spmatrix) -> sp.csr_matrix:
    out = sp.csr_matrix(matrix, copy=True)
    out.data[:] = 1.0
    return out


def drop_heavy_users(inter: Interactions, max_items: int) -> Interactions:
    """Remove visitors who touched more than `max_items` distinct items, then any item left without users."""
    keep_users = np.diff(inter.matrix.indptr) <= max_items
    trimmed = inter.matrix[keep_users]
    keep_items = np.diff(trimmed.tocsc().indptr) > 0
    return Interactions(trimmed[:, keep_items].tocsr(), inter.user_ids[keep_users], inter.item_ids[keep_items])


@dataclass(frozen=True)
class VSM:
    """L2-normalised document-term style matrix: one row per entity, so dot product = cosine."""

    space: str
    matrix: sp.csr_matrix
    ids: np.ndarray

    @property
    def support(self) -> np.ndarray:
        """Number of non-zero entries per row (distinct items for a user, distinct users for an item)."""
        return np.diff(self.matrix.indptr)

    @cached_property
    def transposed(self) -> sp.csr_matrix:
        """Columns-as-rows copy, built once because every neighbour search multiplies by it."""
        return self.matrix.T.tocsr()


def _log_tf(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Sub-linear term frequency: the 50th view of an item says little more than the 5th."""
    out = matrix.copy()
    out.data = np.log1p(out.data)
    return out


def _l2_normalise(matrix: sp.csr_matrix) -> sp.csr_matrix:
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return (sp.diags(1.0 / norms) @ matrix).tocsr()


def log_tf(inter: Interactions) -> sp.csr_matrix:
    return _log_tf(inter.matrix)


def build_vsm(inter: Interactions, space: str, tfidf: bool = True) -> VSM:
    """Build the vector space for users (rows = visitors) or items (rows = items)."""
    if space not in ("user", "item"):
        raise ValueError("space must be 'user' or 'item'")
    tf = _log_tf(inter.matrix)
    if space == "item":
        tf = tf.T.tocsr()
    if tfidf:
        n_rows = tf.shape[0]
        doc_freq = np.diff(tf.tocsc().indptr)
        idf = np.log((1 + n_rows) / (1 + doc_freq)) + 1.0
        tf = (tf @ sp.diags(idf)).tocsr()
    ids = inter.user_ids if space == "user" else inter.item_ids
    return VSM(space, _l2_normalise(tf), ids)


def top_k_neighbours(
    vsm: VSM, query_rows: np.ndarray | list[int], k: int, min_support: int = 1, chunk_size: int = 256
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Top-k cosine neighbours for each query row.

    Returns {row: (neighbour_rows, similarities)} sorted by similarity. The query row itself is
    excluded, and a candidate must have at least `min_support` non-zero entries: with 1 or 2
    data points cosine easily hits 1.0 by coincidence, which is noise rather than similarity.
    """
    matrix, transposed = vsm.matrix, vsm.transposed
    allowed = vsm.support >= min_support
    query_rows = np.asarray(query_rows, dtype=int)
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for start in range(0, len(query_rows), chunk_size):
        block = query_rows[start : start + chunk_size]
        sims = (matrix[block] @ transposed).tocsr()
        for local, row in enumerate(block):
            lo, hi = sims.indptr[local], sims.indptr[local + 1]
            cols, vals = sims.indices[lo:hi], sims.data[lo:hi]
            keep = (cols != row) & allowed[cols]
            cols, vals = cols[keep], vals[keep]
            if len(vals) > k:
                top = np.argpartition(-vals, k - 1)[:k]
                cols, vals = cols[top], vals[top]
            order = np.lexsort((cols, -vals))
            result[int(row)] = (cols[order], vals[order])
    return result


def describe_matrix(matrix: sp.spmatrix) -> dict[str, float]:
    """Shape, sparsity and memory footprint: the 'characteristics' asked for in exam 2.1 no. 4 and 5."""
    matrix = sp.csr_matrix(matrix)
    rows, cols = matrix.shape
    per_row = np.diff(matrix.indptr)
    sparse_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    return {
        "rows": rows,
        "columns": cols,
        "non_zeros": int(matrix.nnz),
        "density_%": 100 * matrix.nnz / (rows * cols),
        "sparsity_%": 100 * (1 - matrix.nnz / (rows * cols)),
        "mean_nnz_per_row": float(per_row.mean()),
        "median_nnz_per_row": float(np.median(per_row)),
        "max_nnz_per_row": int(per_row.max()),
        "sparse_memory_MB": sparse_bytes / 1e6,
        "dense_memory_GB": rows * cols * 8 / 1e9,
    }


def projection_cost(inter: Interactions) -> dict[str, float]:
    """Number of 2-paths that a projection has to enumerate.

    Projecting onto users means pairing every two users that share an item, which costs
    sum_i C(d_i, 2) over item degrees d_i; projecting onto items costs sum_u C(d_u, 2).
    The smaller number is the cheaper projection.
    """
    b = binary(inter.matrix)
    user_deg = np.diff(b.indptr).astype(float)
    item_deg = np.diff(b.tocsc().indptr).astype(float)
    pairs = lambda deg: float((deg * (deg - 1) / 2).sum())
    return {"user_projection_paths": pairs(item_deg), "item_projection_paths": pairs(user_deg)}
