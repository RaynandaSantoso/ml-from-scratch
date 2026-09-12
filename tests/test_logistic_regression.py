import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SklearnLR

from ml_from_scratch.logistic_regression import LogisticRegression


@pytest.fixture
def data():
    """Overlapping (non-separable) classes, so log loss has a finite minimum."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    w_true, b_true = np.array([2.0, -1.0, 0.5]), 0.3
    noise = rng.normal(size=200) * 0.5
    y = (X @ w_true + b_true + noise > 0).astype(float)
    return X, y


@pytest.fixture
def untrained(data):
    """Model at arbitrary non-optimal weights, for cost/gradient checks."""
    X, _ = data
    model = LogisticRegression()
    model.w = np.array([0.3, -0.8, 1.2])
    model.b = 0.4
    return model


@pytest.fixture
def at_zero(data):
    """Model at w=0, b=0 — every predicted probability is exactly 0.5."""
    X, _ = data
    model = LogisticRegression()
    model.w = np.zeros(X.shape[1])
    model.b = 0.0
    return model


# ---------- known-answer tests ----------

def test_cost_at_zero_weights_is_log_two(data, at_zero):
    """w=0,b=0 => every f is sigmoid(0)=0.5 => loss is -log(0.5) per example,
    regardless of the labels. Catches sign errors and a missing /m."""
    X, y = data
    assert at_zero.compute_cost(X, y) == pytest.approx(np.log(2))


def test_cost_at_zero_weights_is_log_two_for_any_labels(data, at_zero):
    X, _ = data
    for y in (np.ones(X.shape[0]), np.zeros(X.shape[0])):
        assert at_zero.compute_cost(X, y) == pytest.approx(np.log(2))


def test_gradient_at_zero_weights_for_all_ones_labels(data, at_zero):
    """All f are 0.5, all y are 1, so err is -0.5 everywhere and db = -0.5."""
    X, _ = data
    _, db = at_zero.compute_gradient(X, np.ones(X.shape[0]))
    assert db == pytest.approx(-0.5)


def test_cost_is_near_zero_for_confident_correct_predictions():
    model = LogisticRegression()
    model.w = np.array([100.0])
    model.b = 0.0
    X = np.array([[1.0], [-1.0]])
    y = np.array([1.0, 0.0])           # both predicted correctly, very confidently
    assert model.compute_cost(X, y) == pytest.approx(0.0, abs=1e-9)


def test_predict_proba_is_sigmoid_of_linear_combination():
    model = LogisticRegression()
    model.w = np.array([1.0, 1.0])
    model.b = 0.0
    # z = 0 -> sigmoid(0) = 0.5
    assert model.predict_proba(np.array([[2.0, -2.0]])) == pytest.approx([0.5])


# ---------- oracle tests ----------

def test_cost_loop_matches_vectorized(data, untrained):
    X, y = data
    assert untrained.compute_cost(X, y) == pytest.approx(
        untrained.compute_cost_loop(X, y)
    )

def test_gradient_loop_matches_vectorized(data, untrained):
    X, y = data
    dw_vec, db_vec = untrained.compute_gradient(X, y)
    dw_loop, db_loop = untrained.compute_gradient_loop(X, y)
    assert dw_vec == pytest.approx(dw_loop)
    assert db_vec == pytest.approx(db_loop)


# ---------- property tests ----------

def test_gradient_matches_finite_differences(data, untrained):
    """Calculus-derived slope must match the measured slope."""
    X, y = data
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


def test_predict_proba_stays_strictly_between_zero_and_one(data):
    X, y = data
    model = LogisticRegression(lr=0.5, n_iters=500).fit(X, y)
    proba = model.predict_proba(X)
    assert ((proba > 0) & (proba < 1)).all()


def test_predict_returns_only_binary_labels(data):
    X, y = data
    model = LogisticRegression(lr=0.5, n_iters=500).fit(X, y)
    assert set(np.unique(model.predict(X))) <= {0, 1}


def test_predict_respects_threshold(data):
    X, y = data
    model = LogisticRegression(lr=0.5, n_iters=500).fit(X, y)
    # a threshold of 0 labels everything positive; 1 labels everything negative
    assert (model.predict(X, threshold=0.0) == 1).all()
    assert (model.predict(X, threshold=1.0) == 0).all()


def test_cost_never_increases_during_training(data):
    X, y = data
    model = LogisticRegression(lr=0.5, n_iters=500).fit(X, y)
    diffs = np.diff(model.cost_history)
    assert (diffs <= 1e-12).all(), "cost went up: learning rate too high, or sign error"


def test_cost_history_has_no_nans(data):
    """Saturated probabilities must not poison the cost via log(0)."""
    X, y = data
    model = LogisticRegression(lr=5.0, n_iters=500).fit(X, y)
    assert not np.isnan(model.cost_history).any()


def test_cost_is_finite_at_extreme_weights(data):
    """Directly exercise the clipping: huge weights saturate the sigmoid."""
    X, y = data
    model = LogisticRegression()
    model.w = np.array([500.0, -500.0, 500.0])
    model.b = 0.0
    cost = model.compute_cost(X, y)
    assert np.isfinite(cost) and not np.isnan(cost)


def test_fit_reduces_cost_and_classifies_well(data):
    X, y = data
    model = LogisticRegression(lr=0.5, n_iters=5000).fit(X, y)
    assert model.cost_history[-1] < model.cost_history[0]
    assert (model.predict(X) == y).mean() > 0.85


def test_fit_agrees_with_sklearn(data):
    """C=np.inf disables sklearn's regularisation, which we don't implement."""
    X, y = data
    mine = LogisticRegression(lr=0.5, n_iters=20000).fit(X, y)
    theirs = SklearnLR(C=np.inf, max_iter=10000).fit(X, y)
    assert mine.w == pytest.approx(theirs.coef_[0], abs=0.05)
    assert mine.b == pytest.approx(theirs.intercept_[0], abs=0.05)


def test_reaches_same_loss_as_sklearn(data):
    """Coefficients can differ slightly while the loss is effectively equal,
    because the surface is nearly flat near the optimum."""
    X, y = data
    mine = LogisticRegression(lr=0.5, n_iters=20000).fit(X, y)
    theirs = SklearnLR(C=np.inf, max_iter=10000).fit(X, y)

    reference = LogisticRegression()
    reference.w, reference.b = theirs.coef_[0], theirs.intercept_[0]
    assert mine.compute_cost(X, y) == pytest.approx(
        reference.compute_cost(X, y), abs=1e-5
    )


def test_fit_returns_self(data):
    X, y = data
    model = LogisticRegression()
    assert model.fit(X, y) is model
