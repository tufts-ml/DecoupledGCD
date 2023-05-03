import pytest
import torch

from dpn.model import DinoCCG


input_shape = (8, 3, 32, 32)
num_classes = 5


class TestDinoCCG():
    @pytest.fixture
    def model(self):
        return DinoCCG(num_classes)

    def test_forward(self, model):
        logits, embeds, means, sigma2s = model(torch.zeros(input_shape))
        assert logits.shape == (input_shape[0], num_classes)
        assert embeds.shape == (input_shape[0], model.embed_len)
        assert means.shape == (num_classes, model.embed_len)
        assert sigma2s.shape == (model.embed_len,)

    def test_anneal_var(self, model):
        # start at init_var
        assert torch.all(model.sigma2s == torch.Tensor([model.init_var] * model.embed_len))
        # middle is average of init and end in 1/x space projected back to x space
        model.anneal_var(model.var_warmup / 2)
        assert torch.all(torch.isclose(
            model.sigma2s,
            torch.Tensor(
                [1 / (((1 / model.init_var) + (1 / model.end_var)) / 2)] * model.embed_len)))
        # end is end var
        model.anneal_var(model.var_warmup)
        assert torch.all(model.sigma2s == torch.Tensor([model.end_var] * model.embed_len))
        # end value used for epochs after milestone
        model.anneal_var(model.var_warmup * 2)
        assert torch.all(model.sigma2s == torch.Tensor([model.end_var] * model.embed_len))
