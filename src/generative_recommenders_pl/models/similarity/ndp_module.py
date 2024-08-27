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

from typing import Optional

import torch


class NDPModule(torch.nn.Module):

    def forward(
        self,
        input_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        item_sideinfo: Optional[torch.Tensor],
        item_ids: torch.Tensor,
        precomputed_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_embeddings: (B, self._input_embedding_dim)
            item_embeddings: (1/B, X, self._item_embedding_dim)
            item_sideinfo: (1/B, X, self._item_sideinfo_dim)
            item_ids: (1/B, X,)
            precomputed_logits: (B, X, self._num_precomputed_logits,)

        Returns:
            Tuple of (B, X,) similarity values, keyed ou\tputs
        """
        pass
