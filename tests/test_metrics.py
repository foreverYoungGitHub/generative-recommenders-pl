import pytest
import torch

from generative_recommenders_pl.models.metrics.retrieval import \
    RetrievalMetrics


@pytest.fixture
def test_data():
    top_k_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    target_ids = torch.tensor([[2], [6], [3]])
    return top_k_ids, target_ids

def test_initialization():
    k = 3
    at_k_list = [1, 2, 3]
    metric = RetrievalMetrics(k=k, at_k_list=at_k_list)
    assert metric.k == k
    assert metric.at_k_list == at_k_list

def test_update_method(test_data):
    top_k_ids, target_ids = test_data
    metric = RetrievalMetrics(k=3, at_k_list=[1, 2, 3])
    metric.update(top_k_ids, target_ids)
    assert len(metric.top_k_ids) == 1
    assert len(metric.target_ids) == 1
    metric.update(top_k_ids, target_ids)
    assert len(metric.top_k_ids) == 2
    assert len(metric.target_ids) == 2

    
def test_compute_method(test_data):
    top_k_ids, target_ids = test_data
    metric = RetrievalMetrics(k=3, at_k_list=[1, 2, 3])
    metric.update(top_k_ids, target_ids)
    output = metric.compute()
    assert output['ndcg@1'] == pytest.approx(0.0, abs=5e-5)
    assert output['ndcg@2'] == pytest.approx(0.2103, abs=5e-5)
    assert output['ndcg@3'] == pytest.approx(0.3770, abs=5e-5)
    assert output['hr@1'] == pytest.approx(0.0, abs=5e-5)
    assert output['hr@2'] == pytest.approx(0.3333, abs=5e-5)
    assert output['hr@3'] == pytest.approx(0.6667, abs=5e-5)
    assert output['mrr'] == pytest.approx(0.3611, abs=5e-5)

def test_reset(test_data):
    top_k_ids, target_ids = test_data
    metric = RetrievalMetrics(k=3, at_k_list=[1, 2, 3])
    metric.update(top_k_ids, target_ids)
    metric.reset()
    assert metric.top_k_ids == []
    assert metric.target_ids == []
