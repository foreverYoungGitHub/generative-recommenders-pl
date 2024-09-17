import torch

from generative_recommenders_pl.models.generative_recommenders import (
    GenerativeRecommenders,
)
from generative_recommenders_pl.models.utils import ops
from generative_recommenders_pl.models.utils.features import (
    SequentialFeatures,
    seq_features_from_row,
)
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)


class Ranking(GenerativeRecommenders):
    def get_ratings_embeddings(self) -> torch.Tensor:
        if not hasattr(self.preprocessor, "ratings_emb"):
            raise ValueError(
                "Preprocessor does not have ratings embeddings, which is required for ranking."
            )
        return self.preprocessor.ratings_emb

    @torch.inference_mode
    def logits(
        self,
        seq_features: SequentialFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the top-k items for the given sequence features.
        """
        seq_embeddings, _ = self.forward(seq_features)  # [B, X]
        current_embeddings = ops.get_current_embeddings(
            seq_features.past_lengths, seq_embeddings
        )

        logits = self.similarity(
            input_embeddings=self.negatives_sampler.normalize_embeddings(
                current_embeddings
            ),  # [N', D]
            item_embeddings=self.negatives_sampler.normalize_embeddings(
                self.get_ratings_embeddings()
            ).unsqueeze(0),  # [1, R, D]
            item_sideinfo=None,
            item_ids=None,
        )[0]  # [N', R]
        return logits

    def ranking(self, batch: tuple[torch.Tensor]) -> torch.Tensor:
        """
        Retrieve the top-k items for the given sequence features.
        """
        # convert the batch to the sequence features (TODO: move to datamodule)
        seq_features, target_ids, target_ratings = seq_features_from_row(
            batch,
            device=self.device,
            max_output_length=self.gr_output_length + 1,
        )

        # embeddings
        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)

        # input features preprocessor
        past_lengths, default_user_embeddings, valid_mask, aux_mask = self.preprocessor(
            past_lengths=seq_features.past_lengths,
            past_ids=seq_features.past_ids,
            past_embeddings=input_embeddings,
            past_payloads=seq_features.past_payloads,
        )

        # forloop the top-k ids and get the user_embeddings
        top_k_ids = seq_features.past_payloads["top_k_ids"]  # B, K
        top_k_item_embeddings = self.embeddings.get_item_embeddings(
            top_k_ids
        )  # B, K, D
        cached_states = None
        for i in range(top_k_ids.size(1)):
            processed_item_embeddings = self.preprocessor.get_processed_item_embeddings(
                top_k_item_embeddings[:, i : i + 1, :]
            )

            # merge processed_item_embeddings with user_embeddings
            # user_embeddings[:, past_lengths:past_lengths+1, :] = processed_item_embeddings
            user_embeddings = (
                default_user_embeddings + processed_item_embeddings
            )  # this is wrong now

            past_lengths += 1

            valid_mask[:, past_lengths : past_lengths + 1] = 1

            # sequence encoder
            user_embeddings, cached_states = self.sequence_encoder(
                past_lengths=past_lengths,
                user_embeddings=user_embeddings,
                valid_mask=valid_mask,
                past_payloads=seq_features.past_payloads,
                cached_states=cached_states,
                return_cache_states=True,
            )

            current_embeddings = ops.get_current_embeddings(
                seq_features.past_lengths, user_embeddings
            )

            logits = self.similarity(
                input_embeddings=self.negatives_sampler.normalize_embeddings(
                    current_embeddings
                ),  # [N', D]
                item_embeddings=self.negatives_sampler.normalize_embeddings(
                    self.get_ratings_embeddings()
                ).unsqueeze(0),  # [1, R, D]
                item_sideinfo=None,
                item_ids=None,
            )[0]
        return logits

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning calls this inside the training loop.

        Args:
            batch (tuple[torch.Tensor]): A tuple containing the input and target
                tensors.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss tensor.
        """
        # convert the batch to the sequence features (TODO: move to datamodule)
        seq_features, target_ids, target_ratings = seq_features_from_row(
            batch,
            device=self.device,
            max_output_length=self.gr_output_length + 1,
        )
        # add target_ids at the end of the past_ids
        seq_features.past_ids.scatter_(
            dim=1,
            index=seq_features.past_lengths.view(-1, 1),
            src=target_ids.view(-1, 1),
        )

        # embeddings
        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
        # TODO: think a better way than replace, since it creates a new instance
        seq_features = seq_features._replace(past_embeddings=input_embeddings)

        # forward pass
        seq_embeddings, _ = self.forward(seq_features)  # [B, X]

        # prepare loss
        supervision_ids = seq_features.past_ids
        supervision_ratings = seq_features.past_payloads["ratings"]
        supervision_ratings.scatter_(
            dim=1,
            index=seq_features.past_lengths.view(-1, 1),
            src=target_ratings.view(-1, 1),
        )

        # dense features to jagged features
        # TODO: seems that the target_ids is not used in the loss
        jagged_features = self.dense_to_jagged(
            lengths=seq_features.past_lengths + 1,
            output_embeddings=seq_embeddings,  # [B, N, D]
            supervision_weights=(supervision_ids != 0).float(),  # ar_mask
            supervision_ratings=supervision_ratings,
        )

        loss = self.loss.jagged_forward(
            negatives_sampler=self.negatives_sampler,
            similarity=self.similarity,
            supervision_embeddings=self.get_ratings_embeddings(),
            **jagged_features,
        )

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        """Lightning calls this at the beginning of the validation epoch."""
        self.metrics.reset()

    def validation_step(
        self, batch: tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Lightning calls this inside the validation loop.

        Args:
            batch (tuple[torch.Tensor]): A tuple containing the input and target
                tensors.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss tensor.
        """
        # convert the batch to the sequence features (TODO: move to datamodule)
        seq_features, target_ids, target_ratings = seq_features_from_row(
            batch,
            device=self.device,
            max_output_length=self.gr_output_length + 1,
        )

        # embeddings
        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
        # TODO: think a better way than replace, since it creates a new instance
        seq_features = seq_features._replace(past_embeddings=input_embeddings)

        # forward pass
        # logits = self.logits(seq_features)
        logits = self.ranking(batch)
        self.metrics.update(preds=logits, target=target_ratings.squeeze(-1))

    def on_validation_epoch_end(self) -> None:
        """Lightning calls this at the end of the validation epoch.

        Args:
            outputs (list[torch.Tensor]): A list of the outputs from each validation step.
        """
        results = self.metrics.compute()
        for k, v in results.items():
            self.log(f"val/{k}", v, on_epoch=True, prog_bar=True, logger=True)
        self.metrics.reset()
        if "monitor" in self.configure_optimizer_params:
            return results[self.configure_optimizer_params["monitor"].split("/")[1]]

    def on_test_epoch_start(self) -> None:
        """Lightning calls this at the beginning of the test epoch."""
        self.metrics.reset()

    def test_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning calls this inside the test loop.

        Args:
            batch (tuple[torch.Tensor]): A tuple containing the input and target
                tensors.
            batch_idx (int): The index of the batch.
        """
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        """Lightning calls this at the end of the test epoch.

        Args:
            outputs (list[torch.Tensor]): A list of the outputs from each test step.
        """
        results = self.metrics.compute()
        for k, v in results.items():
            self.log(f"test/{k}", v, on_epoch=True, prog_bar=True, logger=True)
        self.metrics.reset()
        if "monitor" in self.configure_optimizer_params:
            return results[self.configure_optimizer_params["monitor"].split("/")[1]]

    def predict_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning calls this inside the predict loop.

        Args:
            batch (tuple[torch.Tensor]): A tuple containing the input and target
                tensors.
            batch_idx (int): The index of the batch.
        """
        return self.ranking(batch)
