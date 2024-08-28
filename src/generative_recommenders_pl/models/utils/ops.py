import torch

from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)
__logged_errors = set()

try:
    import fbgemm_gpu  # noqa
except ImportError:
    if "fbgemm_gpu" not in __logged_errors:
        __logged_errors.add("fbgemm_gpu")
        log.error(
            "Failed to import fbgemm_gpu. Falling back to Pytorch implementation."
        )


def asynchronous_complete_cumsum(lengths: torch.Tensor) -> torch.Tensor:
    """
    Args:
        lengths: (B,) x int, where each entry is in [0, X).

    Returns:
        (B,) x int, where each entry is the cumulative sum of the corresponding entry in lengths.
    """
    try:
        return torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    except Exception as e:
        if "asynchronous_complete_cumsum" not in __logged_errors:
            __logged_errors.add("asynchronous_complete_cumsum")
            log.error(f"Error: {e}")
            log.error(
                "Failed to call torch.ops.fbgemm.asynchronous_complete_cumsum. Falling back to Pytorch implementation."
            )

        return torch.cat(
            (torch.tensor([0], dtype=lengths.dtype), torch.cumsum(lengths, dim=0))
        )


def dense_to_jagged(dense_tensor: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """
    Args:
        dense_tensor: (B, N, D,) x float, where B is the batch size, N is the number of elements in the dense tensor, and D is the dimension of each element.
        offsets: (B+1,) x int, where each entry is in [0, N], with an extra offset at the end to define the last slice boundary.

    Returns:
        (X, D,) x float, where X is the corresponding entry in offsets.
    """
    try:
        return torch.ops.fbgemm.dense_to_jagged(dense_tensor, [offsets])[0]
    except Exception as e:
        if "dense_to_jagged" not in __logged_errors:
            __logged_errors.add("dense_to_jagged")
            log.error(f"Error: {e}")
            log.error(
                "Failed to call torch.ops.fbgemm.dense_to_jagged. Falling back to Pytorch implementation."
            )

        jagged_tensors = []
        for i in range(offsets.size(0) - 1):
            length = offsets[i + 1] - offsets[i]
            jagged_tensors.append(dense_tensor[i, :length])
        return torch.cat(jagged_tensors, dim=0)


def jagged_to_padded_dense(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_lengths: int,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Args:
        values: List of (X, D,) x float, where X is the corresponding entry in offsets.
        offsets: (B,) x int, where each entry is in [0, N].
        max_lengths: int, the maximum length of the padded tensor.
        padding_value: float, the value to pad the tensor with.

    Returns:
        (B, max(max_lengths), D,) x float, where each row is a padded tensor from values.
    """
    if not isinstance(max_lengths, int):
        raise ValueError(f"max_lengths must be an integer, but got {type(max_lengths)}")

    try:
        return torch.ops.fbgemm.jagged_to_padded_dense(
            values, [offsets], [max_lengths], padding_value
        )
    except Exception as e:
        if "jagged_to_padded_dense" not in __logged_errors:
            __logged_errors.add("jagged_to_padded_dense")
            log.error(f"Error: {e}")
            log.error(
                "Failed to call torch.ops.fbgemm.jagged_to_padded_dense. Falling back to Pytorch implementation."
            )

        # This implementation is simplified for this specific use case.
        # Calculate the total number of sequences and the maximum sequence length
        num_sequences = offsets.size(0) - 1
        sequences_shape = values.shape[1:]

        # Initialize the padded tensor
        padded_tensor = torch.full(
            (num_sequences, max_lengths, *sequences_shape), padding_value
        )

        # Fill in the padded tensor with values from the jagged tensors
        for i in range(num_sequences):
            start = offsets[i]
            end = offsets[i + 1]
            length = end - start
            padded_tensor[i, :length] = values[start:end]
        return padded_tensor


def batch_gather_embeddings(
    rowwise_indices: torch.Tensor,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        rowwise_indices: (B, N) x int, where each entry is in [0, X).
        embeddings: (B, X, D,) x float.

    Returns:
        (B, N, D,) x float, embeddings corresponding to rowwise_indices.
    """
    _, N = rowwise_indices.size()
    B, X, D = embeddings.size()
    flattened_indices = (
        rowwise_indices
        + torch.arange(
            start=0,
            end=B,
            step=1,
            dtype=rowwise_indices.dtype,
            device=rowwise_indices.device,
        )
        .unsqueeze(1)
        .expand(-1, N)
        * X
    )
    return embeddings.view(-1, D)[flattened_indices, :].reshape(
        rowwise_indices.size() + (D,)
    )


def batch_scatter_embeddings(
    dst_embeddings: torch.Tensor,
    rowwise_indices: torch.Tensor,
    src_embeddings: torch.Tensor,
) -> None:
    """
    Args:
        dst_embeddings: (B, N, D,) x float.
        rowwise_indices: (B,) x int, where each entry is in [0, N - 1).
        source_embeddings: (B, D,) x float.
    """
    B, N, D = dst_embeddings.size()
    flattened_indices = rowwise_indices + torch.arange(
        start=0,
        end=B * N,
        step=N,
        dtype=rowwise_indices.dtype,
        device=rowwise_indices.device,
    )
    dst_embeddings.view(B * N, D)[flattened_indices, :] = src_embeddings


def get_current_embeddings(
    lengths: torch.Tensor,
    encoded_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        lengths: (B,) x int
        seq_embeddings: (B, N, D,) x float

    Returns:
        (B, D,) x float, where [i, :] == encoded_embeddings[i, lengths[i] - 1, :]
    """
    B, N, D = encoded_embeddings.size()
    flattened_offsets = (lengths - 1) + torch.arange(
        start=0, end=B, step=1, dtype=lengths.dtype, device=lengths.device
    ) * N
    return encoded_embeddings.reshape(-1, D)[flattened_offsets, :].reshape(B, D)


def jagged_or_dense_repeat_interleave_dim0(
    x: torch.Tensor, lengths: torch.Tensor, repeats: int
) -> torch.Tensor:
    if len(x.size()) == 3:
        return x.repeat_interleave(repeats, dim=0)
    else:
        assert len(x.size()) == 2, f"x.size() = {x.size()}"
        padded_x = jagged_to_padded_dense(
            values=x,
            offsets=asynchronous_complete_cumsum(lengths),
            max_lengths=lengths.max(),
            padding_value=0.0,
        )
        lengths = lengths.repeat_interleave(repeats, dim=0)
        return dense_to_jagged(
            padded_x.repeat_interleave(repeats, dim=0),
            asynchronous_complete_cumsum(lengths),
        )


def jagged_or_dense_index_select_dim0(
    x: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    if len(x.size()) == 3:
        return x[indices, :, :]
    else:
        assert len(x.size()) == 2, f"x.size() = {x.size()}"
        padded_x = jagged_to_padded_dense(
            values=x,
            offsets=asynchronous_complete_cumsum(lengths),
            max_lengths=lengths.max(),
            padding_value=0.0,
        )
        return dense_to_jagged(
            padded_x[indices, :],
            asynchronous_complete_cumsum(lengths[indices]),
        )
