"""Tests for dynamic capture functionality.

Tests capture_dynamic_execution_graph() and runtime tensor information capture.
"""

import pytest
import torch
import torch.nn as nn
from vode.capture.dynamic_capture import (
    capture_dynamic_execution_graph,
    _tensor_to_tensor_info,
    _count_parameters,
    _module_to_operation_info,
)
from vode.core.nodes import TensorInfo, OperationInfo


class TestTensorToTensorInfo:
    """Test tensor to TensorInfo conversion."""

    def test_simple_tensor(self):
        """Test conversion of simple tensor."""
        tensor = torch.randn(1, 10)
        tensor_info = _tensor_to_tensor_info(tensor, "input")

        assert tensor_info.name == "input"
        assert tensor_info.shape == (1, 10)
        assert tensor_info.dtype == "torch.float32"
        assert tensor_info.device == "cpu"

    def test_multidimensional_tensor(self):
        """Test conversion of multidimensional tensor."""
        tensor = torch.randn(2, 3, 224, 224)
        tensor_info = _tensor_to_tensor_info(tensor, "image")

        assert tensor_info.shape == (2, 3, 224, 224)

    def test_cuda_tensor(self):
        """Test conversion of CUDA tensor (if available)."""
        if torch.cuda.is_available():
            tensor = torch.randn(1, 10).cuda()
            tensor_info = _tensor_to_tensor_info(tensor, "gpu_tensor")

            assert "cuda" in tensor_info.device
        else:
            pytest.skip("CUDA not available")

    def test_different_dtypes(self):
        """Test conversion of tensors with different dtypes."""
        tensor_float = torch.randn(1, 10)
        tensor_int = torch.randint(0, 10, (1, 10))
        tensor_long = torch.randint(0, 10, (1, 10), dtype=torch.long)

        info_float = _tensor_to_tensor_info(tensor_float, "float")
        info_int = _tensor_to_tensor_info(tensor_int, "int")
        info_long = _tensor_to_tensor_info(tensor_long, "long")

        assert "float" in info_float.dtype
        assert "int" in info_int.dtype
        assert "int64" in info_long.dtype or "long" in info_long.dtype


class TestCaptureSimpleModels:
    """Test dynamic capture on simple models."""

    def test_single_linear(self):
        """Test capturing single Linear layer."""
        model = nn.Linear(10, 20)
        x = torch.randn(1, 10)

        root = capture_dynamic_execution_graph(model, x)

        # Node ID is generated, not "root"
        assert root.node_id.startswith("node_")
        assert root.operation.op_type == "Linear"
        assert len(root.inputs) == 1
        assert len(root.outputs) == 1
        assert root.inputs[0].shape == (1, 10)
        assert root.outputs[0].shape == (1, 20)

    def test_simple_sequential(self):
        """Test capturing simple Sequential model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        x = torch.randn(5, 10)

        root = capture_dynamic_execution_graph(model, x)

        assert root.operation.op_type == "Sequential"
        assert len(root.children) == 3

        # Check first layer
        fc1 = root.children[0]
        assert fc1.operation.op_type == "Linear"
        assert len(fc1.inputs) == 1
        assert len(fc1.outputs) == 1
        assert fc1.inputs[0].shape == (5, 10)
        assert fc1.outputs[0].shape == (5, 20)

        # Check ReLU
        relu = root.children[1]
        assert relu.operation.op_type == "ReLU"
        assert relu.inputs[0].shape == (5, 20)
        assert relu.outputs[0].shape == (5, 20)

        # Check second layer
        fc2 = root.children[2]
        assert fc2.operation.op_type == "Linear"
        assert fc2.inputs[0].shape == (5, 20)
        assert fc2.outputs[0].shape == (5, 10)

    def test_conv2d_model(self):
        """Test capturing Conv2d model."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        x = torch.randn(1, 3, 32, 32)

        root = capture_dynamic_execution_graph(model, x)

        # Check Conv2d
        conv = root.children[0]
        assert conv.operation.op_type == "Conv2d"
        assert conv.inputs[0].shape == (1, 3, 32, 32)
        assert conv.outputs[0].shape == (1, 16, 32, 32)

        # Check MaxPool2d
        pool = root.children[2]
        assert pool.operation.op_type == "MaxPool2d"
        assert pool.outputs[0].shape == (1, 16, 16, 16)


class TestCaptureCustomModels:
    """Test dynamic capture on custom models."""

    def test_custom_module(self):
        """Test capturing custom module."""

        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(20, 10)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        model = CustomModel()
        x = torch.randn(2, 10)

        root = capture_dynamic_execution_graph(model, x)

        assert root.operation.op_type == "CustomModel"
        assert len(root.children) == 3
        assert root.inputs[0].shape == (2, 10)
        assert root.outputs[0].shape == (2, 10)

    def test_nested_modules(self):
        """Test capturing nested module structure."""

        class Block(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.fc = nn.Linear(in_features, out_features)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.fc(x))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = Block(10, 20)
                self.block2 = Block(20, 10)

            def forward(self, x):
                x = self.block1(x)
                x = self.block2(x)
                return x

        model = Model()
        x = torch.randn(1, 10)

        root = capture_dynamic_execution_graph(model, x)

        assert root.operation.op_type == "Model"
        assert len(root.children) == 2

        # Check first block
        block1 = root.children[0]
        assert block1.operation.op_type == "Block"
        assert len(block1.children) == 2
        assert block1.inputs[0].shape == (1, 10)
        assert block1.outputs[0].shape == (1, 20)

    def test_residual_connection(self):
        """Test capturing model with residual connections."""

        class ResidualBlock(nn.Module):
            def __init__(self, features):
                super().__init__()
                self.fc = nn.Linear(features, features)
                self.relu = nn.ReLU()

            def forward(self, x):
                residual = x
                x = self.fc(x)
                x = self.relu(x)
                x = x + residual
                return x

        model = ResidualBlock(10)
        x = torch.randn(1, 10)

        root = capture_dynamic_execution_graph(model, x)

        assert root.operation.op_type == "ResidualBlock"
        assert root.inputs[0].shape == (1, 10)
        assert root.outputs[0].shape == (1, 10)


class TestCaptureBatchSizes:
    """Test dynamic capture with different batch sizes."""

    def test_batch_size_1(self):
        """Test with batch size 1."""
        model = nn.Linear(10, 20)
        x = torch.randn(1, 10)

        root = capture_dynamic_execution_graph(model, x)

        assert root.inputs[0].shape == (1, 10)
        assert root.outputs[0].shape == (1, 20)

    def test_batch_size_32(self):
        """Test with batch size 32."""
        model = nn.Linear(10, 20)
        x = torch.randn(32, 10)

        root = capture_dynamic_execution_graph(model, x)

        assert root.inputs[0].shape == (32, 10)
        assert root.outputs[0].shape == (32, 20)

    def test_variable_batch_sizes(self):
        """Test that different batch sizes produce different shapes."""
        model = nn.Linear(10, 20)

        x1 = torch.randn(1, 10)
        root1 = capture_dynamic_execution_graph(model, x1)

        x2 = torch.randn(16, 10)
        root2 = capture_dynamic_execution_graph(model, x2)

        assert root1.inputs[0].shape[0] == 1
        assert root2.inputs[0].shape[0] == 16


class TestCaptureMultipleInputs:
    """Test dynamic capture with multiple inputs."""

    def test_multiple_tensor_inputs(self):
        """Test model with multiple tensor inputs."""

        class MultiInputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(10, 20)

            def forward(self, x1, x2):
                out1 = self.fc1(x1)
                out2 = self.fc2(x2)
                return out1 + out2

        model = MultiInputModel()
        x1 = torch.randn(1, 10)
        x2 = torch.randn(1, 10)

        # Pass as tuple for multiple inputs
        root = capture_dynamic_execution_graph(model, (x1, x2))

        assert root.operation.op_type == "MultiInputModel"
        assert len(root.inputs) == 2
        assert root.inputs[0].shape == (1, 10)
        assert root.inputs[1].shape == (1, 10)


class TestCaptureMultipleOutputs:
    """Test dynamic capture with multiple outputs."""

    def test_tuple_output(self):
        """Test model with tuple output."""

        class MultiOutputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(10, 30)

            def forward(self, x):
                out1 = self.fc1(x)
                out2 = self.fc2(x)
                return out1, out2

        model = MultiOutputModel()
        x = torch.randn(1, 10)

        root = capture_dynamic_execution_graph(model, x)

        assert len(root.outputs) == 2
        assert root.outputs[0].shape == (1, 20)
        assert root.outputs[1].shape == (1, 30)


class TestCaptureEdgeCases:
    """Test dynamic capture edge cases."""

    def test_empty_sequential(self):
        """Test capturing empty Sequential."""
        model = nn.Sequential()
        x = torch.randn(1, 10)

        root = capture_dynamic_execution_graph(model, x)

        assert root.operation.op_type == "Sequential"
        assert len(root.children) == 0

    def test_identity_module(self):
        """Test capturing identity module."""

        class IdentityModel(nn.Module):
            def forward(self, x):
                return x

        model = IdentityModel()
        x = torch.randn(1, 10)

        root = capture_dynamic_execution_graph(model, x)

        assert root.inputs[0].shape == (1, 10)
        assert root.outputs[0].shape == (1, 10)

    def test_invalid_input(self):
        """Test that non-Module input raises TypeError."""
        with pytest.raises(TypeError):
            capture_dynamic_execution_graph("not a module", torch.randn(1, 10))


class TestCaptureParameterInfo:
    """Test that parameter information is captured correctly."""

    def test_parameter_counts(self):
        """Test parameter counting in captured nodes."""
        model = nn.Sequential(
            nn.Linear(10, 20),  # 220 params
            nn.ReLU(),  # 0 params
            nn.Linear(20, 10),  # 210 params
        )
        x = torch.randn(1, 10)

        root = capture_dynamic_execution_graph(model, x)

        # Check root params
        assert root.operation.params_count == 430

        # Check children params
        assert root.children[0].operation.params_count == 220
        assert root.children[1].operation.params_count == 0
        assert root.children[2].operation.params_count == 210

    def test_composite_flag(self):
        """Test that composite flag is set correctly."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Sequential(
                nn.Linear(20, 30),
                nn.ReLU(),
            ),
        )
        x = torch.randn(1, 10)

        root = capture_dynamic_execution_graph(model, x)

        # Root is composite
        assert root.operation.is_composite is True

        # First child is not composite (Linear)
        assert root.children[0].operation.is_composite is False

        # Second child is composite (Sequential)
        assert root.children[1].operation.is_composite is True


class TestCaptureDepthTracking:
    """Test that depth is correctly tracked during capture."""

    def test_depth_levels(self):
        """Test depth tracking across multiple levels."""

        class DeepModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.level1 = nn.Sequential(
                    nn.Sequential(
                        nn.Linear(10, 20),
                    )
                )

            def forward(self, x):
                return self.level1(x)

        model = DeepModel()
        x = torch.randn(1, 10)

        root = capture_dynamic_execution_graph(model, x)

        # Root is at depth 0
        assert root.depth == 0
        # Children depths are based on module hierarchy (name.count("."))
        # level1 has depth 0 (no dots in "level1")
        assert root.children[0].depth == 0
        # level1.0 has depth 1 (one dot)
        assert root.children[0].children[0].depth == 1
        # level1.0.0 has depth 2 (two dots)
        assert root.children[0].children[0].children[0].depth == 2
