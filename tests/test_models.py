import pytest
import torch
from torch.optim import SGD

from ccgaussian.loss import NDCCLoss
from ccgaussian.model import DinoCCG


input_shape = (8, 3, 32, 32)
num_classes = 5


class TestDinoCCG():
    @pytest.fixture
    def model(self):
        return DinoCCG(num_classes)

    def test_forward(self, model):
        logits, norm_embeds, means, sigma2s = model(torch.zeros(input_shape))
        assert logits.shape == (input_shape[0], num_classes)
        assert norm_embeds.shape == (input_shape[0], model.embed_len)
        assert means.shape == (num_classes, model.embed_len)
        assert sigma2s.shape == (model.embed_len,)

    def test_var_stability(self, model):
        norm_embeds = torch.zeros((input_shape[0], model.embed_len))
        norm_embeds[:, :4] = 4
        targets = torch.zeros((input_shape[0],), dtype=int)
        optimizer = SGD([
            {"params": [model.sigma, model.deltas], "lr": 1e-4, "momentum": .9},
            {"params": [model.classifier.weight], "lr": 1e-1, "momentum": .9}])
        # generous loss threshold that should be easy to reach and stay under
        loss_thresh = -400
        passed_loss_thresh = False
        for _ in range(100):
            optimizer.zero_grad()
            means, sigma2s = model.gaussian_params()
            loss_nll = NDCCLoss.nll_loss(norm_embeds, means, sigma2s, targets)
            loss_md = NDCCLoss.md_loss(norm_embeds, means, sigma2s, targets)
            loss = loss_nll + loss_md
            loss.backward()
            optimizer.step()
            if passed_loss_thresh and loss > loss_thresh:
                pytest.fail("Optimization not stable, went under then above threshold")
            if loss < loss_thresh:
                passed_loss_thresh = True
        if passed_loss_thresh is False:
            pytest.fail("Optimization failed to reach threshold")
