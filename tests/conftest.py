import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from unittest.mock import MagicMock, patch


def _make_test_model_file() -> str:
    m = models.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "test_model.pth")
    torch.save(m.state_dict(), path)
    return path


os.environ["MODEL_PATH"] = _make_test_model_file()

_mock_cascade = MagicMock()
_mock_cascade.detectMultiScale.return_value = np.array([[10, 10, 80, 80]])
patch("cv2.CascadeClassifier", return_value=_mock_cascade).start()
