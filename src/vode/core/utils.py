"""Utility functions for VODE core functionality."""

import uuid
from typing import Any


def generate_node_id(prefix: str = "node") -> str:
    """Generate a unique node ID.

    Args:
        prefix: Prefix for the node ID

    Returns:
        Unique node ID string
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def format_shape(shape: tuple[int, ...] | None) -> str:
    """Format a tensor shape for display.

    Args:
        shape: Tensor shape tuple

    Returns:
        Formatted shape string (e.g., "(1, 3, 224, 224)")
    """
    if shape is None:
        return "(?)"
    return f"({', '.join(str(d) for d in shape)})"


def format_dtype(dtype: str | None) -> str:
    """Format a dtype string for display.

    Args:
        dtype: Data type string

    Returns:
        Formatted dtype string
    """
    if dtype is None:
        return "?"

    # Simplify torch dtype names
    if dtype.startswith("torch."):
        return dtype.replace("torch.", "")

    return dtype


def format_device(device: str | None) -> str:
    """Format a device string for display.

    Args:
        device: Device string

    Returns:
        Formatted device string
    """
    if device is None:
        return "?"
    return device


def is_tensor_like(obj: Any) -> bool:
    """Check if an object is tensor-like.

    Args:
        obj: Object to check

    Returns:
        True if object is tensor-like
    """
    # Check for torch.Tensor
    if hasattr(obj, "shape") and hasattr(obj, "dtype") and hasattr(obj, "device"):
        return True

    # Check for numpy array
    if hasattr(obj, "shape") and hasattr(obj, "dtype") and not hasattr(obj, "device"):
        return True

    return False


def get_tensor_info(tensor: Any) -> dict[str, Any]:
    """Extract information from a tensor.

    Args:
        tensor: Tensor object (torch.Tensor or similar)

    Returns:
        Dictionary with tensor information
    """
    info = {
        "shape": None,
        "dtype": None,
        "device": None,
        "requires_grad": False,
    }

    if not is_tensor_like(tensor):
        return info

    # Extract shape
    if hasattr(tensor, "shape"):
        info["shape"] = tuple(tensor.shape)

    # Extract dtype
    if hasattr(tensor, "dtype"):
        info["dtype"] = str(tensor.dtype)

    # Extract device
    if hasattr(tensor, "device"):
        info["device"] = str(tensor.device)

    # Extract requires_grad
    if hasattr(tensor, "requires_grad"):
        info["requires_grad"] = tensor.requires_grad

    return info


def compute_tensor_stats(tensor: Any) -> dict[str, float] | None:
    """Compute statistics for a tensor.

    Args:
        tensor: Tensor object

    Returns:
        Dictionary with statistics or None if not applicable
    """
    if not is_tensor_like(tensor):
        return None

    try:
        # Try to compute statistics
        stats = {}

        if hasattr(tensor, "min"):
            stats["min"] = float(tensor.min())

        if hasattr(tensor, "max"):
            stats["max"] = float(tensor.max())

        if hasattr(tensor, "mean"):
            stats["mean"] = float(tensor.mean())

        if hasattr(tensor, "std"):
            stats["std"] = float(tensor.std())

        return stats if stats else None

    except Exception:
        # If computation fails, return None
        return None


def sanitize_name(name: str) -> str:
    """Sanitize a name for use in node IDs.

    Args:
        name: Original name

    Returns:
        Sanitized name
    """
    # Replace special characters with underscores
    sanitized = name.replace(".", "_").replace("/", "_").replace("\\", "_")
    sanitized = sanitized.replace(" ", "_").replace("-", "_")

    # Remove consecutive underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    return sanitized


def get_module_info(module: Any) -> dict[str, Any]:
    """Extract information from a PyTorch module.

    Args:
        module: nn.Module instance

    Returns:
        Dictionary with module information
    """
    info = {
        "type": type(module).__name__,
        "has_children": False,
        "has_parameters": False,
        "parameter_count": 0,
    }

    # Check if module has children
    if hasattr(module, "children"):
        try:
            children = list(module.children())
            info["has_children"] = len(children) > 0
        except Exception:
            pass

    # Check if module has parameters
    if hasattr(module, "parameters"):
        try:
            params = list(module.parameters())
            info["has_parameters"] = len(params) > 0
            info["parameter_count"] = sum(
                p.numel() for p in params if hasattr(p, "numel")
            )
        except Exception:
            pass

    return info


def truncate_string(s: str, max_length: int = 50) -> str:
    """Truncate a string to a maximum length.

    Args:
        s: String to truncate
        max_length: Maximum length

    Returns:
        Truncated string with ellipsis if needed
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - 3] + "..."
