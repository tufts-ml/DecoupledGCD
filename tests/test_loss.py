import pytest
import torch

import dpn.loss


embed_dim = 768
num_classes = 5
num_labeled_classes = 3
num_unlabeled_classes = num_classes - num_labeled_classes


def dummy_data(num_samples):
    logits = torch.rand((num_samples, num_labeled_classes))
    embeds = torch.rand((num_samples, embed_dim))
    l_proto = torch.rand((num_labeled_classes, embed_dim))
    u_proto = torch.rand((num_unlabeled_classes, embed_dim))
    targets = torch.randint(0, num_labeled_classes, (num_samples,))
    label_mask = torch.tensor(
        ([True] * (num_samples//2)) + ([False] * (num_samples - num_samples//2)),
        dtype=bool)
    uk_mask = ~label_mask
    uk_mask[num_samples - num_samples//2:num_samples - num_samples//4] = False
    return logits, embeds, targets, label_mask, uk_mask, l_proto, u_proto


@pytest.mark.parametrize(
    "loss",
    [
        dpn.loss.DPNLoss(),
    ],
)
class TestDPNLoss():
    def test_empty(self, loss):
        assert loss(*dummy_data(0)) == 0

    def test_random(self, loss):
        assert loss(*dummy_data(8)).shape == torch.Size([])
