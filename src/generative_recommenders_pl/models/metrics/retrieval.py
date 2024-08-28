import torch
import torchmetrics
import torchmetrics.utilities


class RetrievalMetrics(torchmetrics.Metric):
    def __init__(self, k: int, at_k_list: list[int], **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.at_k_list = at_k_list
        self.add_state("top_k_ids", default=[], dist_reduce_fx="cat")
        self.add_state("target_ids", default=[], dist_reduce_fx="cat")

    def update(self, top_k_ids: torch.Tensor, target_ids: torch.Tensor, **kwargs):
        self.top_k_ids.append(top_k_ids)
        self.target_ids.append(target_ids)

    def compute(self):
        # Concatenate the lists of tensors
        top_k_ids = torchmetrics.utilities.dim_zero_cat(self.top_k_ids)
        target_ids = torchmetrics.utilities.dim_zero_cat(self.target_ids)

        assert top_k_ids.size(1) == self.k
        _, rank_indices = torch.max(
            torch.cat(
                [top_k_ids, target_ids],
                dim=1,
            )
            == target_ids,
            dim=1,
        )
        ranks = rank_indices + 1
        output = {}
        # compute ndcg
        for at_k in self.at_k_list:
            output[f"ndcg@{at_k}"] = torch.where(
                ranks <= at_k,
                1.0 / torch.log2(ranks + 1),
                torch.zeros(1, dtype=torch.float32, device=ranks.device),
            ).mean()
        # compute recall / hit rate
        for at_k in self.at_k_list:
            output[f"hr@{at_k}"] = (ranks <= at_k).to(torch.float32).mean()
        # compute mrr
        output["mrr"] = (1.0 / ranks).mean()
        return output
