import pytest
import torch

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

    def test_anneal_var(self, model):
        assert torch.all(model.sigma2s == torch.Tensor([model.init_var] * model.embed_len))
        model.anneal_var(model.var_milestone / 2)
        assert torch.all(
            model.sigma2s == torch.Tensor([(model.init_var + model.end_var) / 2] * model.embed_len))
        model.anneal_var(model.var_milestone)
        assert torch.all(model.sigma2s == torch.Tensor([model.end_var] * model.embed_len))
        model.anneal_var(model.var_milestone * 2)
        assert torch.all(model.sigma2s == torch.Tensor([model.end_var] * model.embed_len))
