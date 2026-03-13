"""Base tracer class for VODE capture mechanisms.

Defines the abstract interface that all tracers must implement.
"""

from abc import ABC, abstractmethod
from typing import Any
import torch.nn as nn

from vode.core import ExecutionNode


class BaseTracer(ABC):
    """Abstract base class for all VODE tracers.

    All tracers follow the same pattern:
    1. Initialize with a model
    2. Capture execution (static or dynamic)
    3. Return an ExecutionNode

    Attributes:
        model: PyTorch model being traced
    """

    def __init__(self, model: nn.Module):
        """Initialize tracer with a model.

        Args:
            model: PyTorch model to trace

        Raises:
            TypeError: If model is not an nn.Module
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(model).__name__}")

        self.model = model

    @abstractmethod
    def capture(self, *args: Any, **kwargs: Any) -> ExecutionNode:
        """Capture model execution.

        Args:
            *args: Positional arguments for capture
            **kwargs: Keyword arguments for capture

        Returns:
            ExecutionNode with captured data
        """
        pass
