import pytest
import torch

from ccgaussian.loss import NDCCLoss


embed_dim = 768
num_classes = 5


class TestNDCCLoss():
    @pytest.fixture
    def loss(self):
        return NDCCLoss(w_nll=.05)

    def test_empty(self, loss):
        logits = torch.empty((0, num_classes))
        norm_embeds = torch.empty((0, embed_dim))
        means = torch.zeros((num_classes, embed_dim))
        sigma2s = torch.ones(embed_dim)
        targets = torch.empty((0,), dtype=torch.int64)
        assert loss(logits, norm_embeds, means, sigma2s, targets) == 0
