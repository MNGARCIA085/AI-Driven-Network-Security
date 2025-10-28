import pytest
import torch
import torch.nn as nn
from src.models.nnet import NNModel


def test_nn_model_forward():
    input_size = 4
    num_classes = 3
    batch_size = 5

    model = NNModel(input_size=input_size, num_classes=num_classes)

    # Input tensor
    x = torch.rand(batch_size, input_size)
    
    # Forward pass
    out = model(x)
    
    # Check output shape
    assert out.shape == (batch_size, num_classes)
    
    # Check output type
    assert isinstance(out, torch.Tensor)
    
    # Check model is a subclass of nn.Module
    assert isinstance(model, nn.Module)
