import numpy as np
import pytest

from ml_from_scratch.activations import relu, sigmoid, softmax


# ---------- relu ----------

def test_relu_zeroes_negatives_and_keeps_positives():
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert relu(z) == pytest.approx([0.0, 0.0, 0.0, 0.5, 2.0])


def test_relu_works_on_matrices():
    """dense() calls g on a whole (m, units) matrix, not one number at a time."""
    Z = np.array([[-1.0, 3.0], [4.0, -5.0]])
    np.testing.assert_array_equal(relu(Z), [[0.0, 3.0], [4.0, 0.0]])


# ---------- softmax: known answers ----------

def test_softmax_of_equal_logits_is_uniform():
    assert softmax(np.zeros(4)) == pytest.approx([0.25] * 4)


def test_softmax_matches_definition():
    z = np.array([1.0, 2.0, 3.0])
    expected = np.exp(z) / np.exp(z).sum()
    assert softmax(z) == pytest.approx(expected)


def test_two_class_softmax_is_sigmoid_of_the_difference():
    """softmax([0, z])[1] = e^z / (1 + e^z) = sigmoid(z).
    This is why softmax regression generalises logistic regression."""
    for z in (-3.0, 0.0, 0.7, 4.0):
        assert softmax(np.array([0.0, z]))[1] == pytest.approx(sigmoid(z))


# ---------- softmax: properties ----------

def test_softmax_outputs_are_probabilities():
    z = np.random.default_rng(0).normal(size=10)
    a = softmax(z)
    assert (a > 0).all()
    assert a.sum() == pytest.approx(1.0)


def test_softmax_preserves_shape():
    Z = np.random.default_rng(0).normal(size=(5, 3))
    assert softmax(Z).shape == (5, 3)


def test_softmax_normalises_each_row_separately():
    """For a batch (m, K), each example is its own distribution: every row
    sums to 1, not the whole matrix."""
    Z = np.random.default_rng(1).normal(size=(5, 3))
    A = softmax(Z)
    assert A.sum(axis=1) == pytest.approx(np.ones(5))
    for i in range(Z.shape[0]):
        assert A[i] == pytest.approx(softmax(Z[i]))


def test_softmax_is_unchanged_by_adding_a_constant():
    z = np.array([0.5, -1.0, 2.0])
    assert softmax(z + 100.0) == pytest.approx(softmax(z))


def test_softmax_preserves_order():
    z = np.array([0.1, 3.0, -2.0, 1.0])
    assert np.argmax(softmax(z)) == np.argmax(z)


# ---------- softmax: numerical stability ----------

def test_softmax_handles_large_logits():
    """Naive np.exp(1000) overflows to inf and inf/inf is nan."""
    a = softmax(np.array([1000.0, 1001.0]))
    assert not np.isnan(a).any()
    assert a == pytest.approx(softmax(np.array([0.0, 1.0])))


def test_softmax_handles_large_logits_in_a_batch():
    """The stability trick must shift each row by its own max."""
    Z = np.array([[1000.0, 1001.0], [-1000.0, -999.0], [0.0, 1.0]])
    A = softmax(Z)
    assert not np.isnan(A).any()
    for row in A:
        assert row == pytest.approx(softmax(np.array([0.0, 1.0])))


def test_softmax_handles_extreme_spread():
    a = softmax(np.array([-1000.0, 1000.0]))
    assert not np.isnan(a).any()
    assert a == pytest.approx([0.0, 1.0])


def test_softmax_does_not_modify_its_input():
    """An in-place `z -= z.max()` would silently change the caller's array."""
    z = np.array([1.0, 2.0, 3.0])
    softmax(z)
    np.testing.assert_array_equal(z, [1.0, 2.0, 3.0])
