import abc

import torch
import torch.nn.functional as F

from generative_recommenders_pl.models.negatives_samples.negative_sampler import \
    NegativesSampler
from generative_recommenders_pl.models.similarity.ndp_module import NDPModule


class AutoregressiveLoss(torch.nn.Module):

    @abc.abstractmethod
    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Variant of forward() when the tensors are already in jagged format.

        Args:
            output_embeddings: [N', D] x float, embeddings for the current
                input sequence.
            supervision_ids: [N'] x int64, (positive) supervision ids.
            supervision_embeddings: [N', D] x float.
            supervision_weights: Optional [N'] x float. Optional weights for
                masking out invalid positions, or reweighting supervision labels.
            negatives_sampler: sampler used to obtain negative examples paired with
                positives.

        Returns:
            (1), loss for the current engaged sequence.
        """
        pass



class BCELoss(AutoregressiveLoss):
    def __init__(
        self,
        temperature: float,
        model,  #: NDPModule,
    ) -> None:
        super().__init__()
        self._temperature: float = temperature
        self._model = model

    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
        similarity: NDPModule,
    ) -> torch.Tensor:
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        assert supervision_ids.size() == supervision_weights.size()

        sampled_ids, sampled_negative_embeddings = negatives_sampler(
            positive_ids=supervision_ids,
            num_to_sample=1,
        )

        supervision_embeddings = negatives_sampler.normalize_embeddings(
            supervision_embeddings
        )
        positive_logits = (
            similarity(
                input_embeddings=output_embeddings,  # [B, D] = [N', D]
                item_embeddings=supervision_embeddings.unsqueeze(
                    1
                ),  # [N', D] -> [N', 1, D]
                item_sideinfo=None,
                item_ids=supervision_ids.unsqueeze(1), # [N', 1]
                precomputed_logits=None,
            )[0].squeeze(1) / self._temperature
        )  # [N']

        sampled_negatives_logits = (
            similarity(
                input_embeddings=output_embeddings,  # [N', D]
                item_embeddings=sampled_negative_embeddings,  # [N', 1, D]
                item_sideinfo=None,
                item_ids=sampled_ids,  # [N', 1]
                precomputed_logits=None,
            )[0].squeeze(1) / self._temperature
        )  # [N']
        sampled_negatives_valid_mask = (
            supervision_ids != sampled_ids.squeeze(1)
        ).float()  # [N']
        loss_weights = supervision_weights * sampled_negatives_valid_mask
        weighted_losses = (
            (
                F.binary_cross_entropy_with_logits(
                    input=positive_logits,
                    target=torch.ones_like(positive_logits),
                    reduction="none",
                )
                + F.binary_cross_entropy_with_logits(
                    input=sampled_negatives_logits,
                    target=torch.zeros_like(sampled_negatives_logits),
                    reduction="none",
                )
            )
            * loss_weights
            * 0.5
        )
        return weighted_losses.sum() / loss_weights.sum()


class BCELossWithRatings(AutoregressiveLoss):
    def __init__(
        self,
        temperature: float,
        model,  #: NDPModule,
    ) -> None:
        super().__init__()
        self._temperature: float = temperature
        self._model = model

    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        supervision_ratings: torch.Tensor,
        negatives_sampler: NegativesSampler,
        similarity: NDPModule,
    ) -> torch.Tensor:
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        assert supervision_ids.size() == supervision_weights.size()

        supervision_embeddings = negatives_sampler.normalize_embeddings(
            supervision_embeddings
        )
        target_logits = (
            similarity(
                input_embeddings=output_embeddings,  # [B, D] = [N', D]
                item_embeddings=supervision_embeddings.unsqueeze(
                    1
                ),  # [N', D] -> [N', 1, D]
                item_sideinfo=None,
                item_ids=supervision_ids.unsqueeze(1), # [N', 1]
                precomputed_logits=None,
            )[0].squeeze(1) / self._temperature
        )  # [N']

        # loss_weights = (supervision_ids > 0).to(torch.float32)

        weighted_losses = (
            F.binary_cross_entropy_with_logits(
                input=target_logits,
                target=supervision_ratings.to(dtype=target_logits.dtype),
                reduction="none",
            )
        ) * supervision_weights
        return weighted_losses.sum() / supervision_weights.sum()


class SampledSoftmaxLoss(AutoregressiveLoss):

    def __init__(
        self,
        num_to_sample: int,
        softmax_temperature: float,
    ) -> None:
        super().__init__()
        self._num_to_sample: int = num_to_sample
        self._softmax_temperature: float = softmax_temperature

    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
        similarity: NDPModule,
    ) -> torch.Tensor:
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        assert supervision_ids.size() == supervision_weights.size()

        sampled_ids, sampled_negative_embeddings = negatives_sampler(
            positive_ids=supervision_ids,
            num_to_sample=self._num_to_sample,
        )
        positive_embeddings = negatives_sampler.normalize_embeddings(
            supervision_embeddings
        )
        positive_logits = (
            similarity(
                input_embeddings=output_embeddings,  # [B, D] = [N', D]
                item_embeddings=positive_embeddings.unsqueeze(
                    1
                ),  # [N', D] -> [N', 1, D]
                item_sideinfo=None,
                item_ids=supervision_ids.unsqueeze(1), # [N', 1]
                precomputed_logits=None,
            )
            / self._softmax_temperature
        )  # [N', 1]
        sampled_negatives_logits = similarity(
            input_embeddings=output_embeddings,  # [N', D]
            item_embeddings=sampled_negative_embeddings,  # [N', R, D]
            item_sideinfo=None,
            item_ids=sampled_ids,  # [N', R]
            precomputed_logits=None,
        ) # [N', R]
        sampled_negatives_logits = torch.where(
            supervision_ids.unsqueeze(1) == sampled_ids,  # [N', R]
            -5e4,
            sampled_negatives_logits / self._softmax_temperature,
        )
        jagged_loss = -F.log_softmax(
            torch.cat([positive_logits, sampled_negatives_logits], dim=1), dim=1
        )[:, 0]
        return (jagged_loss * supervision_weights).sum() / supervision_weights.sum()
