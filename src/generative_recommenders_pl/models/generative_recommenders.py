from typing import Any

import hydra
import lightning as L
import torch
import torchmetrics
from omegaconf import DictConfig

from generative_recommenders_pl.data.reco_dataset import RecoDataModule
from generative_recommenders_pl.models.embeddings.embeddings import EmbeddingModule
from generative_recommenders_pl.models.indexing.candidate_index import CandidateIndex
from generative_recommenders_pl.models.losses.autoregressive_losses import (
    AutoregressiveLoss,
)
from generative_recommenders_pl.models.negatives_samples.negative_sampler import (
    InBatchNegativesSampler,
    NegativesSampler,
)
from generative_recommenders_pl.models.postprocessors.postprocessors import (
    OutputPostprocessorModule,
)
from generative_recommenders_pl.models.preprocessors.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders_pl.models.similarity.ndp_module import NDPModule
from generative_recommenders_pl.models.utils import ops
from generative_recommenders_pl.models.utils.features import (
    SequentialFeatures,
    seq_features_from_row,
)
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)


class GenerativeRecommenders(L.LightningModule):
    def __init__(
        self,
        datamodule: RecoDataModule | DictConfig,
        embeddings: EmbeddingModule | DictConfig,
        preprocessor: InputFeaturesPreprocessorModule | DictConfig,
        sequence_encoder: torch.nn.Module | DictConfig,
        postprocessor: OutputPostprocessorModule | DictConfig,
        similarity: NDPModule | DictConfig,
        negatives_sampler: NegativesSampler | DictConfig,
        candidate_index: CandidateIndex | DictConfig,
        loss: AutoregressiveLoss | DictConfig,
        metrics: torchmetrics.Metric | DictConfig,
        optimizer: torch.optim.Optimizer | DictConfig,
        scheduler: torch.optim.lr_scheduler.LRScheduler | DictConfig,
        configure_optimizer_params: DictConfig,
        gr_output_length: int,
        item_embedding_dim: int,
        compile_model: bool,
    ) -> None:
        super().__init__()

        self.optimizer: torch.optim.Optimizer = (
            hydra.utils.instantiate(optimizer)
            if isinstance(optimizer, DictConfig)
            else optimizer
        )
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = (
            hydra.utils.instantiate(scheduler)
            if isinstance(scheduler, DictConfig)
            else scheduler
        )
        self.configure_optimizer_params: dict[str, Any] = configure_optimizer_params

        self.gr_output_length: int = gr_output_length
        self.item_embedding_dim: int = item_embedding_dim
        self.compile_model: bool = compile_model

        self.__hydra_init_submodules(
            datamodule=datamodule,
            embeddings=embeddings,
            preprocessor=preprocessor,
            sequence_encoder=sequence_encoder,
            postprocessor=postprocessor,
            similarity=similarity,
            negatives_sampler=negatives_sampler,
            candidate_index=candidate_index,
            loss=loss,
            metrics=metrics,
        )

    def __hydra_init_submodules(
        self,
        datamodule: RecoDataModule,
        embeddings: EmbeddingModule | DictConfig,
        preprocessor: InputFeaturesPreprocessorModule | DictConfig,
        sequence_encoder: torch.nn.Module | DictConfig,
        postprocessor: OutputPostprocessorModule | DictConfig,
        similarity: NDPModule | DictConfig,
        negatives_sampler: NegativesSampler | DictConfig,
        candidate_index: CandidateIndex | DictConfig,
        loss: AutoregressiveLoss | DictConfig,
        metrics: torchmetrics.Metric | DictConfig,
    ) -> None:
        def init_embedding_module(embeddings: EmbeddingModule) -> EmbeddingModule:
            if isinstance(embeddings, DictConfig):
                kwargs = {}
                if "num_items" not in embeddings:
                    kwargs["num_items"] = datamodule.max_item_id
                if "item_embedding_dim" not in embeddings:
                    kwargs["item_embedding_dim"] = self.item_embedding_dim
                return hydra.utils.instantiate(embeddings, **kwargs)
            else:
                return embeddings

        def init_preprocessor_module(
            preprocessor: InputFeaturesPreprocessorModule | DictConfig,
        ) -> InputFeaturesPreprocessorModule:
            if isinstance(embeddings, DictConfig):
                kwargs = {}
                if "max_sequence_len" not in preprocessor:
                    kwargs["max_sequence_len"] = (
                        datamodule.max_sequence_length + self.gr_output_length + 1
                    )
                if "embedding_dim" not in preprocessor:
                    kwargs["embedding_dim"] = self.item_embedding_dim
                return hydra.utils.instantiate(preprocessor, **kwargs)
            else:
                return preprocessor

        def init_sequence_encoder_module(
            sequence_encoder: torch.nn.Module | DictConfig,
        ) -> torch.nn.Module:
            if isinstance(sequence_encoder, DictConfig):
                kwargs = {}
                if "max_sequence_len" not in sequence_encoder:
                    kwargs["max_sequence_len"] = datamodule.max_sequence_length
                if "max_output_len" not in sequence_encoder:
                    kwargs["max_output_len"] = self.gr_output_length + 1
                if "embedding_dim" not in sequence_encoder:
                    kwargs["embedding_dim"] = self.item_embedding_dim
                if "item_embedding_dim" not in sequence_encoder:
                    kwargs["item_embedding_dim"] = self.item_embedding_dim
                if "attention_dim" not in sequence_encoder:
                    kwargs["attention_dim"] = self.item_embedding_dim
                if "linear_dim" not in sequence_encoder:
                    kwargs["linear_dim"] = self.item_embedding_dim
                return hydra.utils.instantiate(sequence_encoder, **kwargs)
            else:
                return sequence_encoder

        def init_postprocessor_module(
            postprocessor: OutputPostprocessorModule | DictConfig,
        ) -> OutputPostprocessorModule:
            if isinstance(postprocessor, DictConfig):
                kwargs = {}
                if "embedding_dim" not in postprocessor:
                    kwargs["embedding_dim"] = self.item_embedding_dim
                return hydra.utils.instantiate(postprocessor, **kwargs)
            else:
                return postprocessor

        def init_similarity_module(similarity: NDPModule | DictConfig) -> NDPModule:
            if isinstance(similarity, DictConfig):
                return hydra.utils.instantiate(similarity)
            else:
                return similarity

        def init_negatives_sampler_module(
            negatives_sampler: NegativesSampler | DictConfig,
        ) -> NegativesSampler:
            if isinstance(negatives_sampler, DictConfig):
                kwargs = {}
                if negatives_sampler["_target_"].endswith("LocalNegativesSampler"):
                    if "num_items" not in negatives_sampler:
                        kwargs["all_item_ids"] = datamodule.all_item_ids
                return hydra.utils.instantiate(negatives_sampler, **kwargs)
            else:
                return negatives_sampler

        def init_candidate_index_module(
            candidate_index: CandidateIndex,
        ) -> CandidateIndex:
            if isinstance(candidate_index, DictConfig):
                kwargs = {}
                if "ids" not in candidate_index:
                    kwargs["ids"] = datamodule.all_item_ids
                return hydra.utils.instantiate(candidate_index, **kwargs)
            else:
                return candidate_index

        def init_loss_module(
            loss: AutoregressiveLoss | DictConfig,
        ) -> AutoregressiveLoss:
            if isinstance(loss, DictConfig):
                return hydra.utils.instantiate(loss)
            else:
                return loss

        def init_metrics_module(
            metrics: torchmetrics.Metric | DictConfig,
        ) -> torchmetrics.Metric:
            if isinstance(metrics, DictConfig):
                return hydra.utils.instantiate(metrics)
            else:
                return metrics

        self.embeddings: EmbeddingModule = init_embedding_module(embeddings)
        self.preprocessor: InputFeaturesPreprocessorModule = init_preprocessor_module(
            preprocessor
        )
        self.sequence_encoder: torch.nn.Module = init_sequence_encoder_module(
            sequence_encoder
        )
        self.postprocessor: OutputPostprocessorModule = init_postprocessor_module(
            postprocessor
        )
        self.similarity: NDPModule = init_similarity_module(similarity)
        self.negatives_sampler: NegativesSampler = init_negatives_sampler_module(
            negatives_sampler
        )
        self.candidate_index: CandidateIndex = init_candidate_index_module(
            candidate_index
        )
        self.loss: AutoregressiveLoss = init_loss_module(loss)
        self.metrics: torchmetrics.Metric = init_metrics_module(metrics)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        Args:
            stage (str): One of 'fit', 'validate', 'test', or 'predict'.
        """
        if self.compile_model and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            dict[str, Any]: A dict containing the configured optimizers and learning-rate
                schedulers to be used for training.
        """
        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **self.configure_optimizer_params,
                },
            }
        return {"optimizer": optimizer}

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # Call the superclass's state_dict method to get the full state dictionary
        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        # List of module names you don't want to save
        modules_to_exclude = [
            "similarity",
            "negatives_sampler",
            "candidate_index",
            "loss",
            "metrics",
        ]

        # Remove the keys corresponding to the modules to exclude
        keys_to_remove = [
            key
            for key in state_dict.keys()
            for module_name in modules_to_exclude
            if key.startswith(prefix + module_name)
        ]
        for key in keys_to_remove:
            del state_dict[key]

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        # since we removed some keys from the state_dict, we need to set strict=False
        super().load_state_dict(state_dict, strict=False)

    def forward(
        self, seq_features: SequentialFeatures
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lightning calls this inside the training loop.

        Args:
            seq_features (SequentialFeatures): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
            cached_states: The cached states.
        """
        # input features preprocessor
        past_lengths, user_embeddings, valid_mask = self.preprocessor(
            past_lengths=seq_features.past_lengths,
            past_ids=seq_features.past_ids,
            past_embeddings=seq_features.past_embeddings,
            past_payloads=seq_features.past_payloads,
        )

        # sequence encoder
        user_embeddings, cached_states = self.sequence_encoder(
            past_lengths=past_lengths,
            user_embeddings=user_embeddings,
            valid_mask=valid_mask,
            past_payloads=seq_features.past_payloads,
        )

        # output postprocessor
        encoded_embeddings = self.postprocessor(user_embeddings)
        return encoded_embeddings, cached_states

    @torch.inference_mode
    def retrieve(
        self,
        seq_features: SequentialFeatures,
        filter_past_ids: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the top-k items for the given sequence features.

        """
        seq_embeddings, _ = self.forward(seq_features)  # [B, X]
        current_embeddings = ops.get_current_embeddings(
            seq_features.past_lengths, seq_embeddings
        )

        if self.candidate_index.embeddings is None:
            log.info(
                "Initializing candidate index embeddings with current item embeddings"
            )
            self.candidate_index.update_embeddings(
                self.negatives_sampler.normalize_embeddings(
                    self.embeddings.get_item_embeddings(self.candidate_index.ids)
                )
            )

        top_k_ids, top_k_scores = self.candidate_index.get_top_k_outputs(
            query_embeddings=current_embeddings,
            invalid_ids=(seq_features.past_ids if filter_past_ids else None),
        )
        return top_k_ids, top_k_scores

    def dense_to_jagged(
        self, lengths: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor]:
        """Convert dense tensor to jagged tensor.

        Args:
            lengths (torch.Tensor): The lengths tensor.
            **kwargs: The dict with the dense tensor to be converted.

        Returns:
            dict[str, torch.Tensor]: The jagged tensor.
        """
        jagged_id_offsets = ops.asynchronous_complete_cumsum(lengths)
        return {
            key: ops.dense_to_jagged(kwargs[key], jagged_id_offsets)
            if key != "supervision_ids"
            else ops.dense_to_jagged(
                kwargs[key].unsqueeze(-1).float(), jagged_id_offsets
            )
            .squeeze(1)
            .long()
            for key in kwargs
        }

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

        # negative sampling
        if isinstance(self.negatives_sampler, InBatchNegativesSampler):
            # get_item_embeddings currently assume 1-d tensor.
            in_batch_ids = supervision_ids.view(-1)
            self.negatives_sampler.process_batch(
                ids=in_batch_ids,
                presences=(in_batch_ids != 0),
                embeddings=self.embeddings.get_item_embeddings(in_batch_ids),
            )
        else:
            # update embedding in the  local negative sampling
            self.negatives_sampler._item_emb = self.embeddings._item_emb

        # dense features to jagged features
        # TODO: seems that the target_ids is not used in the loss
        jagged_features = self.dense_to_jagged(
            lengths=seq_features.past_lengths,
            output_embeddings=seq_embeddings[:, :-1, :],  # [B, N-1, D]
            supervision_ids=supervision_ids[:, 1:],  # [B, N-1]
            supervision_embeddings=input_embeddings[:, 1:, :],  # [B, N - 1, D]
            supervision_weights=(supervision_ids[:, 1:] != 0).float(),  # ar_mask
        )

        loss = self.loss.jagged_forward(
            negatives_sampler=self.negatives_sampler,
            similarity=self.similarity,
            **jagged_features,
        )

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        """Lightning calls this at the beginning of the validation epoch."""
        self.metrics.reset()
        self.candidate_index.update_embeddings(
            self.negatives_sampler.normalize_embeddings(
                self.embeddings.get_item_embeddings(self.candidate_index.ids)
            )
        )

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
        top_k_ids, top_k_scores = self.retrieve(seq_features)
        self.metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)

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
        self.candidate_index.update_embeddings(
            self.negatives_sampler.normalize_embeddings(
                self.embeddings.get_item_embeddings(self.candidate_index.ids)
            )
        )

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
