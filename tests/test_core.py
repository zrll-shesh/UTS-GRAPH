import numpy as np
import pandas as pd
import pytest

from src import metrics, vsm


@pytest.fixture
def inter():
    edges = pd.DataFrame(
        {
            "visitorId": [1, 1, 2, 2, 3, 3, 3, 4],
            "itemId": [10, 20, 10, 30, 20, 30, 40, 40],
            "rel": ["VIEWED", "BOUGHT", "VIEWED", "VIEWED", "VIEWED", "ADDED_TO_CART", "VIEWED", "VIEWED"],
            "n": [2, 1, 1, 3, 1, 1, 1, 1],
        }
    )
    return vsm.build_interactions(edges, {"VIEWED": 1.0, "ADDED_TO_CART": 3.0, "BOUGHT": 5.0})


def test_interaction_weights(inter):
    row, col = inter.user_index[1], inter.item_index[10]
    assert inter.matrix[row, col] == 2.0
    assert inter.matrix[inter.user_index[1], inter.item_index[20]] == 5.0


@pytest.mark.parametrize("space", ["user", "item"])
def test_top_k_matches_dense_cosine(inter, space):
    model = vsm.build_vsm(inter, space)
    dense = model.matrix.toarray()
    full = dense @ dense.T
    found = vsm.top_k_neighbours(model, np.arange(len(model.ids)), k=2)
    for row, (cols, sims) in found.items():
        expected = np.delete(full[row], row)
        assert np.allclose(sorted(sims, reverse=True), sorted(expected[expected > 0], reverse=True)[: len(sims)])
        assert row not in cols


def test_rows_are_unit_length(inter):
    norms = np.sqrt(vsm.build_vsm(inter, "user").matrix.multiply(vsm.build_vsm(inter, "user").matrix).sum(axis=1))
    assert np.allclose(np.asarray(norms).ravel(), 1.0)


def test_drop_heavy_users(inter):
    trimmed = vsm.drop_heavy_users(inter, max_items=2)
    assert 3 not in trimmed.user_ids


def test_hit_and_ndcg():
    ranked, relevant = [5, 6, 7, 8, 9], {7, 99}
    assert metrics.hit_at_n(ranked, relevant, 5) == 1.0
    assert metrics.hit_at_n(ranked, relevant, 2) == 0.0
    expected = (1 / np.log2(4)) / (1 / np.log2(2) + 1 / np.log2(3))
    assert metrics.ndcg_at_n(ranked, relevant, 5) == pytest.approx(expected)
    assert metrics.ndcg_at_n([7, 99], {7, 99}, 5) == pytest.approx(1.0)


def test_projection_cost_counts_two_paths(inter):
    cost = vsm.projection_cost(inter)
    assert cost["user_projection_paths"] == 4
    assert cost["item_projection_paths"] == 5
