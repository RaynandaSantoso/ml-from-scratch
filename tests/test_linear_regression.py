import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLR

from ml_from_scratch.linear_regression import LinearRegression


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    w_true, b_true = np.array([2.0, -1.0, 0.5]), 0.7
    y = X @ w_true + b_true + rng.normal(size=50) * 0.1
    return X, y, w_true, b_true


@pytest.fixture
def untrained(data):
    """A model with arbitrary (non-optimal) weights, for gradient/cost checks."""
    X, y, _, _ = data
    model = LinearRegression()
    model.w = np.array([0.3, -0.8, 1.2])
    model.b = 0.4
    return model


# ---------- 1. known-answer tests ----------

def test_predict_computes_wx_plus_b():
    model = LinearRegression()
    model.w = np.array([1.0, 1.0])
    model.b = 0.0
    # 1*2 + 1*3 + 0 = 5
    assert model.predict(np.array([[2.0, 3.0]])) == pytest.approx([5.0])


def test_predict_handles_multiple_rows():
    model = LinearRegression()
    model.w = np.array([2.0, 0.0])
    model.b = 1.0
    X = np.array([[1.0, 99.0], [3.0, -99.0]])
    assert model.predict(X) == pytest.approx([3.0, 7.0])


def test_cost_is_zero_for_perfect_predictions():
    model = LinearRegression()
    model.w = np.array([2.0])
    model.b = 1.0
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 3.0, 5.0])          # exactly 2x + 1
    assert model.compute_cost(X, y) == pytest.approx(0.0)


def test_cost_matches_hand_computed_value():
    model = LinearRegression()
    model.w = np.array([0.0])
    model.b = 0.0                          # predicts 0 for everything
    X = np.array([[0.0], [0.0]])
    y = np.array([3.0, 4.0])
    # errors are -3 and -4; sum of squares = 25; m = 2; 25 / (2*2) = 6.25
    assert model.compute_cost(X, y) == pytest.approx(6.25)


# ---------- 2. oracle tests: loop vs vectorized ----------

def test_cost_loop_matches_vectorized(data, untrained):
    X, y, _, _ = data
    assert untrained.compute_cost(X, y) == pytest.approx(
        untrained.compute_cost_loop(X, y)
    )


def test_gradient_loop_matches_vectorized(data, untrained):
    X, y, _, _ = data
    dw_vec, db_vec = untrained.compute_gradient(X, y)
    dw_loop, db_loop = untrained.compute_gradient_loop(X, y)
    assert dw_vec == pytest.approx(dw_loop)
    assert db_vec == pytest.approx(db_loop)


# ---------- 3. property tests ----------

def test_gradient_matches_finite_differences(data, untrained):
    """The slope we derive by calculus must match the slope we measure."""
    X, y, _, _ = data
    dw_analytic, db_analytic = untrained.compute_gradient(X, y)
    eps = 1e-6

    dw_numeric = np.zeros_like(untrained.w)
    for j in range(untrained.w.size):
        original = untrained.w[j]
        untrained.w[j] = original + eps
        cost_hi = untrained.compute_cost(X, y)
        untrained.w[j] = original - eps
        cost_lo = untrained.compute_cost(X, y)
        untrained.w[j] = original
        dw_numeric[j] = (cost_hi - cost_lo) / (2 * eps)

    assert dw_analytic == pytest.approx(dw_numeric, abs=1e-6)

    original = untrained.b
    untrained.b = original + eps
    cost_hi = untrained.compute_cost(X, y)
    untrained.b = original - eps
    cost_lo = untrained.compute_cost(X, y)
    untrained.b = original
    assert db_analytic == pytest.approx((cost_hi - cost_lo) / (2 * eps), abs=1e-6)


def test_cost_never_increases_during_training(data):
    X, y, _, _ = data
    model = LinearRegression(lr=0.1, n_iters=200).fit(X, y)
    diffs = np.diff(model.cost_history)
    assert (diffs <= 1e-12).all(), "cost went up: learning rate too high, or sign error"


def test_fit_recovers_known_coefficients(data):
    X, y, w_true, b_true = data
    model = LinearRegression(lr=0.1, n_iters=5000).fit(X, y)
    assert model.w == pytest.approx(w_true, abs=0.05)
    assert model.b == pytest.approx(b_true, abs=0.05)


def test_fit_agrees_with_sklearn(data):
    X, y, _, _ = data
    mine = LinearRegression(lr=0.1, n_iters=5000).fit(X, y)
    theirs = SklearnLR().fit(X, y)
    assert mine.w == pytest.approx(theirs.coef_, abs=1e-3)
    assert mine.b == pytest.approx(theirs.intercept_, abs=1e-3)


def test_fit_returns_self(data):
    X, y, _, _ = data
    model = LinearRegression()
    assert model.fit(X, y) is model
