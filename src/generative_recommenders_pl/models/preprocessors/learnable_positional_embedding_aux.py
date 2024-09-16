import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from generative_recommenders_pl.models.preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders_pl.models.utils.initialization import (
    init_mlp_xavier_weights_zero_bias,
    truncated_normal,
)


class OneHotEncoding(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return torch.nn.functional.one_hot(x, num_classes=self.num_classes).float()


class LearnablePositionalEmbeddingAuxInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):
    def __init__(
        self,
        max_sequence_len: int,
        embedding_dim: int,
        dropout_rate: float,
        auxiliary_columns: list[str],
        encoding_type: list[str],
        categorical_dim: list[int],
        categorical_embedding_dim: list[int],
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
        self._max_sequence_len: int = max_sequence_len
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)

        self._auxiliary_columns = auxiliary_columns
        self._encoding_type = encoding_type
        self._categorical_dim = categorical_dim
        self._categorical_embedding_dim = categorical_embedding_dim

        self.make_auxiliary_projections()
        self.reset_state()

    def make_auxiliary_projections(self):
        if len(self._auxiliary_columns) == 0:
            raise ValueError("No auxiliary columns found")
        if len(self._encoding_type) != len(self._auxiliary_columns):
            raise ValueError(
                "The length of auxiliary columns and encoding type must be the same"
            )
        if len(self._categorical_dim) != len(self._auxiliary_columns):
            raise ValueError(
                "The length of auxiliary columns and categorical dimension must be the same"
            )
        if len(self._categorical_embedding_dim) != len(self._auxiliary_columns):
            raise ValueError(
                "The length of auxiliary columns and categorical embedding dimension must be the same"
            )

        aux_input_dim = 0
        for name, encoding_type, categorical_dim, categorical_embedding_dim in zip(
            self._auxiliary_columns,
            self._encoding_type,
            self._categorical_dim,
            self._categorical_embedding_dim,
        ):
            if encoding_type == "onehot":
                setattr(
                    self,
                    f"_aux_proj_{name}",
                    OneHotEncoding(num_classes=categorical_dim),
                )
                init_mlp_xavier_weights_zero_bias(getattr(self, f"_aux_proj_{name}"))
                aux_input_dim += categorical_dim
            elif encoding_type == "embed":
                setattr(
                    self,
                    f"_aux_proj_{name}",
                    torch.nn.Embedding(
                        num_embeddings=categorical_dim,
                        embedding_dim=categorical_embedding_dim,
                    ),
                )
                truncated_normal(
                    getattr(self, f"_aux_proj_{name}").weight.data,
                    mean=0.0,
                    std=math.sqrt(1.0 / categorical_embedding_dim),
                )
                aux_input_dim += categorical_embedding_dim
            elif encoding_type == "numeric":
                setattr(self, f"_aux_proj_{name}", torch.nn.Identity())
                aux_input_dim += 1
            else:
                raise ValueError(f"Invalid encoding type: {encoding_type}")
        self._aux_projection = torch.nn.Linear(aux_input_dim, self._embedding_dim)
        init_mlp_xavier_weights_zero_bias(self._aux_projection)

    def debug_str(self) -> str:
        return f"posi_aux_d{self._dropout_rate}"

    def reset_state(self):
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        B, N = past_ids.size()

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]

        # merge aux embedding into past_embeddings
        past_lengths = past_lengths + 1
        if past_lengths.max() > self._max_sequence_len:
            raise ValueError(
                f"past_lengths exceed max_sequence_len: {past_lengths.max()}, max_sequence_len: {self._max_sequence_len}"
            )

        # concatenate auxiliary embeddings to past_embeddings at the beginning of the sequence
        aux_embeddings = F.normalize(
            (
                self._aux_projection(
                    torch.cat(
                        [
                            getattr(self, f"_aux_proj_{col}")(past_payloads.pop(col))
                            for col in self._auxiliary_columns
                        ],
                        dim=-1,
                    )
                )
            ),
            p=2,
            dim=-1,
        )  # [B, 1, D]
        past_embeddings = torch.cat(
            [aux_embeddings.unsqueeze(1), past_embeddings * (self._embedding_dim**0.5)],
            dim=1,
        )  # [B, N+1, D]
        valid_mask = torch.cat(
            [torch.ones_like(valid_mask[:, :1]), valid_mask], dim=1
        )  # [B, N+1]
        past_payloads["ratings"] = torch.cat(
            [
                -torch.ones_like(past_payloads["ratings"][:, :1]),
                past_payloads["ratings"],
            ],
            dim=1,
        )  # [B, N+1]
        past_payloads["timestamps"] = torch.cat(
            [
                torch.zeros_like(past_payloads["timestamps"][:, :1]),
                past_payloads["timestamps"],
            ],
            dim=1,
        )  # [B, N+1]

        # remove the last element of past_embeddings and valid_mask and all sequtential features in past_payloads
        past_embeddings = past_embeddings[:, :-1, :]  # [B, N, D]
        valid_mask = valid_mask[:, :-1]  # [B, N, 1]
        # since the dict is mutable, the past_payloads has updated in the original dict
        past_payloads["ratings"] = past_payloads["ratings"][:, :-1]  # [B, N]
        past_payloads["timestamps"] = past_payloads["timestamps"][:, :-1]  # [B, N]

        # generate user_embeddings finally
        user_embeddings = past_embeddings + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)
        user_embeddings *= valid_mask

        # set aux_mask to 0 for the first item
        aux_mask = torch.arange(N, device=past_ids.device).unsqueeze(
            0
        ) < past_lengths.unsqueeze(1)
        aux_mask[:, 0] = 0
        return past_lengths, user_embeddings, valid_mask, aux_mask
