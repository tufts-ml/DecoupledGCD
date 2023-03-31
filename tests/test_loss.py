import pytest
import torch

import ccgaussian.loss


embed_dim = 768
num_classes = 5


def dummy_data(num_samples):
    logits = torch.rand((num_samples, num_classes))
    embeds = torch.rand((num_samples, embed_dim))
    means = torch.rand((num_classes, embed_dim))
    sigma2s = torch.rand(embed_dim) + 1
    targets = torch.randint(0, num_classes, (num_samples,))
    return logits, embeds, means, sigma2s, targets


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
        logits, embeds, means, sigma2s, targets = dummy_data(0)
        assert loss(logits, embeds, means, sigma2s, targets) == 0


def test_soft_consistent():
    loss = ccgaussian.loss.NDCCFixedLoss(w_nll=.05)
    soft_loss = ccgaussian.loss.NDCCFixedSoftLoss(w_nll=.05)
    num_samples = 8
    logits, embeds, means, sigma2s, targets = dummy_data(num_samples)
    soft_targets = torch.zeros((num_samples, num_classes))
    soft_targets[torch.arange(num_samples), targets] = 1
    loss_val = loss(logits, embeds, means, sigma2s, targets)
    soft_loss_val = soft_loss(logits, embeds, means, sigma2s, soft_targets)
    assert torch.allclose(loss_val, soft_loss_val)
