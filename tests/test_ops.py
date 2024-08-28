import torch

from generative_recommenders_pl.models.utils import ops


def test_asynchronous_complete_cumsum():
    # Setup
    lengths = torch.Tensor([1, 2]).int()

    # Expected output
    expected_output = torch.Tensor([0, 1, 3]).int()

    # Execution
    result = ops.asynchronous_complete_cumsum(lengths)

    # Verification
    assert result.shape == expected_output.shape, "Shape mismatch"
    assert torch.all(torch.eq(result, expected_output)), "Content mismatch"


def test_dense_to_jagged():
    # Setup
    lengths = torch.Tensor([1, 2]).int()
    x_offsets = ops.asynchronous_complete_cumsum(lengths)
    x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    x = x.unsqueeze(-1)

    # Expected output
    expected_output = torch.Tensor([[1], [4], [5]])

    # Execution
    result = ops.dense_to_jagged(x, x_offsets)

    # Verification
    assert result.shape == expected_output.shape, "Shape mismatch"
    assert torch.all(torch.eq(result, expected_output)), "Content mismatch"


def test_jagged_to_padded_dense():
    # Setup
    values = torch.Tensor([1, 4, 5]).unsqueeze(-1)
    offsets = torch.tensor([0, 1, 3])

    # Expected output
    expected_output = torch.Tensor([[1, 0, 0], [4, 5, 0]]).unsqueeze(-1)

    # Execution
    result = ops.jagged_to_padded_dense(values, offsets, 3, 0)

    # Verification
    assert result.shape == expected_output.shape, "Shape mismatch"
    assert torch.all(torch.eq(result, expected_output)), "Content mismatch"
