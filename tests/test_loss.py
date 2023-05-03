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
    norm_mask = torch.tensor([True] * num_samples)
    return logits, embeds, means, sigma2s, targets, norm_mask


@pytest.mark.parametrize(
    "loss",
    [
        ccgaussian.loss.NDCCLoss(w_nll=.025),
        ccgaussian.loss.NDCCFixedLoss(w_nll=.025),
    ],
)
class TestNDCCLoss():
    def test_empty(self, loss):
        logits, embeds, means, sigma2s, targets, _ = dummy_data(0)
        assert loss(logits, embeds, means, sigma2s, targets) == 0


def test_empty_soft():
    loss = ccgaussian.loss.GMMFixedLoss(w_nll=.025, w_unlab=0, pseudo_thresh=0)
    logits, embeds, means, sigma2s, targets, norm_mask = dummy_data(0)
    assert loss(logits, embeds, means, sigma2s, targets, norm_mask) == 0


def test_soft_consistent():
    loss = ccgaussian.loss.NDCCFixedLoss(w_nll=.025)
    soft_loss = ccgaussian.loss.GMMFixedLoss(w_nll=.025, w_unlab=0, pseudo_thresh=0)
    num_samples = 8
    logits, embeds, means, sigma2s, targets, norm_mask = dummy_data(num_samples)
    soft_targets = torch.zeros((num_samples, num_classes))
    soft_targets[torch.arange(num_samples), targets] = 1
    loss_val = loss(logits, embeds, means, sigma2s, targets)
    soft_loss_val = soft_loss(logits, embeds, means, sigma2s, soft_targets, norm_mask)
    assert torch.allclose(loss_val, soft_loss_val)
