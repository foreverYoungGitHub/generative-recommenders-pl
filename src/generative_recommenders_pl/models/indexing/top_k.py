# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Tuple

import torch

class TopKModule(torch.nn.Module):

    @abc.abstractmethod
    def forward(
        self,
        query_embeddings: torch.Tensor,
        item_embeddings_t: torch.Tensor,
        item_ids: torch.Tensor,
        k: int,
        sorted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, X, ...). Implementation-specific.
            k: int. top k to return.
            sorted: bool.

        Returns:
            Tuple of (top_k_scores, top_k_ids), both of shape (B, K,)
        """
        pass


class MIPSBruteForceTopK(TopKModule):
    def forward(
        self,
        query_embeddings: torch.Tensor,
        item_embeddings_t: torch.Tensor,
        item_ids: torch.Tensor,
        k: int,
        sorted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, ...). Implementation-specific.
            k: int. final top-k to return.
            sorted: bool. whether to sort final top-k results or not.

        Returns:
            Tuple of (top_k_scores x float, top_k_ids x int), both of shape (B, K,)
        """
        # (B, X,)
        all_logits = torch.mm(query_embeddings, item_embeddings_t)
        top_k_logits, top_k_indices = torch.topk(
            all_logits, dim=1, k=k, sorted=sorted, largest=True,
        )  # (B, k,)
        return top_k_logits, item_ids.squeeze(0)[top_k_indices]
