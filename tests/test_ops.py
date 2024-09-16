import pytest
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mask_dense_by_aux_mask(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Test case 1: Basic functionality
    dense_tensor = torch.tensor(
        [[[1, 1], [2, 2], [3, 3], [4, 4]], [[5, 5], [6, 6], [7, 7], [8, 8]]],
        dtype=torch.float,
        device=device,
    )
    aux_mask = torch.tensor(
        [[False, True, False, True], [True, False, True, False]], device=device
    )
    lengths = torch.tensor([4, 4], device=device)

    expected_output = torch.tensor(
        [[[2, 2], [4, 4], [0, 0], [0, 0]], [[5, 5], [7, 7], [0, 0], [0, 0]]],
        dtype=torch.float,
        device=device,
    )
    expected_new_lengths = torch.tensor([2, 2], device=device)

    output, new_lengths = ops.mask_dense_by_aux_mask(
        dense_tensor, aux_mask, lengths, dense_tensor.shape[1]
    )
    assert torch.allclose(output, expected_output)
    assert torch.all(new_lengths == expected_new_lengths)

    # Test case 2: Different lengths
    dense_tensor = torch.tensor(
        [[[1, 1], [2, 2], [3, 3], [4, 4]], [[5, 5], [6, 6], [0, 0], [0, 0]]],
        dtype=torch.float,
        device=device,
    )
    aux_mask = torch.tensor(
        [[False, True, False, True], [True, False, False, False]], device=device
    )
    lengths = torch.tensor([4, 2], device=device)

    expected_output = torch.tensor(
        [[[2, 2], [4, 4], [0, 0], [0, 0]], [[5, 5], [0, 0], [0, 0], [0, 0]]],
        dtype=torch.float,
        device=device,
    )
    expected_new_lengths = torch.tensor([2, 1], device=device)

    output, new_lengths = ops.mask_dense_by_aux_mask(
        dense_tensor, aux_mask, lengths, dense_tensor.shape[1]
    )
    assert torch.allclose(output, expected_output)
    assert torch.all(new_lengths == expected_new_lengths)

    # Test case 3: All masked
    dense_tensor = torch.tensor(
        [[[1, 1], [2, 2]], [[3, 3], [4, 4]]], dtype=torch.float, device=device
    )
    aux_mask = torch.tensor([[False, False], [False, False]], device=device)
    lengths = torch.tensor([2, 2], device=device)

    expected_output = torch.zeros_like(dense_tensor)
    expected_new_lengths = torch.zeros_like(lengths)

    output, new_lengths = ops.mask_dense_by_aux_mask(
        dense_tensor, aux_mask, lengths, dense_tensor.shape[1]
    )
    assert torch.allclose(output, expected_output)
    assert torch.all(new_lengths == expected_new_lengths)

    # Test case 4: None masked
    dense_tensor = torch.tensor(
        [[[1, 1], [2, 2]], [[3, 3], [4, 4]]], dtype=torch.float, device=device
    )
    aux_mask = torch.tensor([[True, True], [True, True]], device=device)
    lengths = torch.tensor([2, 2], device=device)

    expected_output = dense_tensor.clone()
    expected_new_lengths = lengths.clone()

    output, new_lengths = ops.mask_dense_by_aux_mask(
        dense_tensor, aux_mask, lengths, dense_tensor.shape[1]
    )
    assert torch.allclose(output, expected_output)
    assert torch.all(new_lengths == expected_new_lengths)
