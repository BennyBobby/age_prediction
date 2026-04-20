import torch
import torch.nn as nn
from src.model import get_model


def test_output_shape():
    model = get_model()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1)


def test_output_is_single_value_per_sample():
    model = get_model()
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 1)


def test_classifier_head_replaced():
    model = get_model()
    head = model.classifier[2]
    assert isinstance(head, nn.Linear)
    assert head.out_features == 1


def test_no_activation_on_output():
    # Age regression: final layer must be Linear with no activation
    model = get_model()
    assert isinstance(model.classifier[2], nn.Linear)
    # classifier has exactly 3 elements: LayerNorm, Flatten, Linear
    assert len(model.classifier) == 3
