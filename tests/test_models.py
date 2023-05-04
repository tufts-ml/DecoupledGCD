import pytest
import torch

from dpn.model import DPN


input_shape = (8, 3, 32, 32)
embed_len = 768

num_classes = 5
num_labeled_classes = 3
num_unlabeled_classes = num_classes - num_labeled_classes


class TestDPN():
    @pytest.fixture
    def model(self):
        l_proto = torch.zeros((num_labeled_classes, embed_len))
        u_proto = torch.zeros((num_unlabeled_classes, embed_len))
        return DPN(num_classes, num_labeled_classes, l_proto, u_proto)

    def test_forward(self, model):
        logits, embeds = model(torch.zeros(input_shape))
        assert logits.shape == (input_shape[0], num_labeled_classes)
        assert embeds.shape == (input_shape[0], model.embed_len)
