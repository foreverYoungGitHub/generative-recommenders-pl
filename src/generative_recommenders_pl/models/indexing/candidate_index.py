from typing import Optional, Tuple

import torch

from generative_recommenders_pl.models.indexing.top_k import TopKModule
from generative_recommenders_pl.models.utils import ops


class CandidateIndex(object):

    def __init__(
        self,
        k: int,
        ids: torch.Tensor,
        top_k_module: TopKModule,
        embeddings: torch.Tensor = None,
        invalid_ids: Optional[torch.Tensor] = None,
        debug_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._ids: torch.Tensor = torch.as_tensor(ids).unsqueeze(0)
        self._k = min(k, self._ids.shape[1])
        self._top_k_module: TopKModule = top_k_module
        self._invalid_ids: Optional[torch.Tensor] = invalid_ids
        self._debug_path: Optional[str] = debug_path
        self.update_embeddings(embeddings)

    def update_embeddings(self, embeddings: torch.Tensor) -> None:
        if embeddings is not None:
            self._embeddings_t = embeddings.permute(2, 1, 0).squeeze(2)
        else:
            self._embeddings_t = None

    @property
    def ids(self) -> torch.Tensor:
        """
        Returns:
            (1, X) or (B, X), where valid ids are positive integers.
        """
        return self._ids

    @property
    def num_objects(self) -> int:
        return self._ids.size(1)

    @property
    def embeddings(self) -> torch.Tensor:
        """
        Returns:
            (1, X, D) or (B, X, D) with the same shape as `ids'.
        """
        return self._embeddings_t.unsqueeze(2).permute(2, 1, 0).squeeze(2)

    def filter_invalid_ids(
        self,
        invalid_ids: torch.Tensor,
    ) -> "CandidateIndex":
        """
        Filters invalid_ids (batch dimension dependent) from the current index.

        Args:
            invalid_ids: (B, N) x int64.

        Returns:
            CandidateIndex with invalid_ids filtered.
        """
        if self._ids.size(0) == 1:
            # ((1, X, 1) == (B, 1, N)) -> (B, X)
            invalid_mask, _ = (self._ids.unsqueeze(2) == invalid_ids.unsqueeze(1)).max(
                dim=2
            )
            lengths = (~invalid_mask).int().sum(-1)  # (B,)
            valid_1d_mask = (~invalid_mask).view(-1)
            B: int = lengths.size(0)
            D: int = self._embeddings_t.size(0)
            jagged_ids = self._ids.expand(B, -1).reshape(-1)[valid_1d_mask]
            jagged_embeddings = self.embeddings.expand(B, -1, -1).reshape(-1, D)[
                valid_1d_mask
            ]
            X_prime: int = lengths.max(-1)[0].item()
            jagged_offsets = ops.asynchronous_complete_cumsum(lengths)
            return CandidateIndex(
                ids=ops.jagged_to_padded_dense(
                    values=jagged_ids.unsqueeze(-1),
                    offsets=[jagged_offsets],
                    max_lengths=[X_prime],
                    padding_value=0,
                ).squeeze(-1),
                embeddings=ops.jagged_to_padded_dense(
                    values=jagged_embeddings,
                    offsets=[jagged_offsets],
                    max_lengths=[X_prime],
                    padding_value=0.0,
                ),
                debug_path=self._debug_path,
            )
        else:
            assert self._invalid_ids is None
            return CandidateIndex(
                k=self._k,
                ids=self.ids,
                top_k_module=self._top_k_module,
                embeddings=self.embeddings,
                invalid_ids=invalid_ids,
                debug_path=self._debug_path,
            )

    def get_top_k_outputs(
        self,
        query_embeddings: torch.Tensor,
        k: int = None,
        invalid_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Gets top-k outputs specified by `top_k_module', while filtering out
        invalid ids per row as specified by `invalid_ids'.

        Args:
            query_embeddings: (B * r, ...). Implementation-specific.
            k: int. top k to return.
            invalid_ids: (B * r, N_0) x int64. The list of ids (if > 0) to filter from
                results if present. Expect N_0 to be a small constant.
        Returns:
            A tuple of (top_k_ids, top_k_prs) of shape (B * r, k, ...).
        """
        max_num_invalid_ids = 0
        if invalid_ids is not None:
            max_num_invalid_ids = invalid_ids.size(1)

        if k is None:
            k = self._k

        k_prime = min(k + max_num_invalid_ids, self.num_objects)
        top_k_prime_scores, top_k_prime_ids = self._top_k_module(
            query_embeddings=query_embeddings,
            item_embeddings_t=self._embeddings_t,
            item_ids=self._ids,
            k=k_prime,
            sorted=True,
        )

        # Masks out invalid items rowwise.
        if invalid_ids is not None:
            id_is_valid = ~(
                (top_k_prime_ids.unsqueeze(2) == invalid_ids.unsqueeze(1)).max(2)[0]
            )  # [B, K + N_0]
            id_is_valid = torch.logical_and(
                id_is_valid, torch.cumsum(id_is_valid.int(), dim=1) <= k
            )
            # [[1, 0, 1, 0], [0, 1, 1, 1]], k=2 -> [[0, 2], [1, 2]]
            top_k_rowwise_offsets = torch.nonzero(id_is_valid, as_tuple=True)[1].view(
                -1, k
            )
            top_k_scores = torch.gather(
                top_k_prime_scores, dim=1, index=top_k_rowwise_offsets
            )
            top_k_ids = torch.gather(
                top_k_prime_ids, dim=1, index=top_k_rowwise_offsets
            )
        else:
            # id_is_valid = torch.ones_like(top_k_prime_indices, dtype=torch.bool, device=expanded_ids.device)
            top_k_scores = top_k_prime_scores
            top_k_ids = top_k_prime_ids

        return top_k_ids, top_k_scores

    def apply_object_filter(self) -> "CandidateIndex":
        """
        Applies general per batch filters.
        """
        raise NotImplementedError("not implemented.")
