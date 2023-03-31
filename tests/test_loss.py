import pytest
import torch

import ccgaussian.loss


embed_dim = 768
num_classes = 5


@pytest.mark.parametrize(
    "loss",
    [
        ccgaussian.loss.NDCCLoss(w_nll=.05),
        ccgaussian.loss.NDCCFixedLoss(w_nll=.05),
        ccgaussian.loss.NDCCFixedSoftLoss(w_nll=.05),
    ],
)
class TestNDCCLoss():
    def test_empty(self, loss):
        logits = torch.empty((0, num_classes))
        norm_embeds = torch.empty((0, embed_dim))
        means = torch.zeros((num_classes, embed_dim))
        sigma2s = torch.ones(embed_dim)
        targets = torch.empty((0,), dtype=torch.int64)
        assert loss(logits, norm_embeds, means, sigma2s, targets) == 0
