import torch

from fraud_sequence_transformer.application.sequence_models import MODELS


def test_every_model_gives_one_logit_per_window():
    vocab = [5, 3, 7]
    tokens = torch.randint(0, 3, (4, 10, 3))
    for name, cls in MODELS.items():
        assert cls(vocab)(tokens).shape == (4,), name
