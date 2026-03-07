"""Value extraction system for function-level tracing.

This module processes function parameters and return values captured during tracing,
extracting metadata, statistics, and previews for various types including torch tensors.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from vode.trace.models import (
    TensorMeta,
    TensorStats,
    TensorValuePolicy,
    ValuePreview,
    VariableRecord,
)


class ValueExtractor:
    """Extracts and processes values from function boundaries.

    This class handles extraction of parameters, return values, and local variables,
    with special support for torch tensors and nested data structures.
    """

    def __init__(self):
        """Initialize the value extractor."""
        # Try to import torch
        self._torch_available = False
        self._torch_tensor_class = None
        self._torch_parameter_class = None
        self._torch_module_class = None
        self._try_import_torch()

    def _try_import_torch(self) -> None:
        """Attempt to import torch and cache relevant classes."""
        try:
            import torch
            import torch.nn as nn

            self._torch_available = True
            self._torch_tensor_class = torch.Tensor
            self._torch_parameter_class = nn.Parameter
            self._torch_module_class = nn.Module
        except ImportError:
            self._torch_available = False

    def extract_parameters(
        self, frame: Any, func: Callable | None, config: Any
    ) -> dict[str, VariableRecord]:
        """Extract function parameters from frame.

        Args:
            frame: Execution frame containing locals
            func: Function object (if available) for signature inspection
            config: TraceConfig with value extraction policy

        Returns:
            Dictionary mapping parameter names to VariableRecord objects
        """
        parameters: dict[str, VariableRecord] = {}

        # Get function signature if possible
        param_names = []
        if func is not None:
            try:
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
            except (ValueError, TypeError):
                # Signature not available, fall back to heuristics
                pass

        # If signature not available, try to infer from frame locals
        if not param_names:
            # Get code object to determine parameter count
            code = frame.f_code
            param_names = list(code.co_varnames[: code.co_argcount])

        # Extract each parameter value
        frame_locals = frame.f_locals
        for param_name in param_names:
            if param_name in frame_locals:
                value = frame_locals[param_name]
                slot_path = f"arg.{param_name}"

                var_record = self._process_value(value, param_name, slot_path, config)
                parameters[param_name] = var_record

        return parameters

    def extract_return_value(
        self, value: Any, config: Any
    ) -> VariableRecord | dict[str, VariableRecord] | None:
        """Extract return value and standardize it.

        Args:
            value: The return value from the function
            config: TraceConfig with value extraction policy

        Returns:
            VariableRecord for single values, dict for tuple/dict returns, None for None
        """
        if value is None:
            return None

        # Standardize the return value structure
        standardized = self._standardize_value(value)

        # If it's a simple value, return single VariableRecord
        if not isinstance(standardized, dict):
            return self._process_value(value, "return", "return", config)

        # If it's a structured return (tuple/list/dict), create records for each slot
        result: dict[str, VariableRecord] = {}
        for slot_path, slot_value in standardized.items():
            display_name = slot_path.split(".")[-1]  # Get last part of path
            var_record = self._process_value(
                slot_value, display_name, slot_path, config
            )
            result[slot_path] = var_record

        return result

    def extract_locals(self, frame: Any, config: Any) -> dict[str, VariableRecord]:
        """Extract local variables from frame (optional, for debugging).

        Args:
            frame: Execution frame containing locals
            config: TraceConfig with value extraction policy

        Returns:
            Dictionary mapping variable names to VariableRecord objects
        """
        locals_dict: dict[str, VariableRecord] = {}

        if not config.capture_locals:
            return locals_dict

        frame_locals = frame.f_locals

        for var_name, value in frame_locals.items():
            # Skip special variables
            if var_name.startswith("__"):
                continue

            slot_path = f"local.{var_name}"
            var_record = self._process_value(value, var_name, slot_path, config)
            locals_dict[var_name] = var_record

        return locals_dict

    def _process_value(
        self, value: Any, name: str, slot_path: str, config: Any
    ) -> VariableRecord:
        """Process a value and create a VariableRecord.

        Args:
            value: The value to process
            name: Display name for the value
            slot_path: Slot path (e.g., 'arg.input', 'return.0')
            config: TraceConfig with value extraction policy

        Returns:
            VariableRecord containing extracted metadata
        """
        runtime_id = id(value)
        type_name = type(value).__qualname__

        # Add module prefix for better type identification
        type_module = type(value).__module__
        if type_module and type_module not in ("builtins", "__main__"):
            type_name = f"{type_module}.{type_name}"

        # Initialize fields
        tensor_meta = None
        tensor_stats = None
        preview = None

        # Check if this is a torch tensor
        if self._torch_available and isinstance(value, self._torch_tensor_class):
            tensor_meta = self._extract_tensor_meta(value)
            tensor_stats = self._compute_tensor_stats(value, config.value_policy)
            preview = self._create_tensor_preview(value, config.value_policy)

        # Check if this is a torch.nn.Parameter
        elif self._torch_available and isinstance(value, self._torch_parameter_class):
            tensor_meta = self._extract_tensor_meta(value)
            tensor_stats = self._compute_tensor_stats(value, config.value_policy)
            preview = self._create_tensor_preview(value, config.value_policy)
            type_name = "torch.nn.Parameter"

        # Check if this is a torch.nn.Module
        elif self._torch_available and isinstance(value, self._torch_module_class):
            # Don't recurse into modules, just record type
            preview = ValuePreview(text=f"<{type(value).__name__} module>", data=None)

        # Handle primitive types
        elif isinstance(value, (int, float, str, bool, type(None))):
            preview = self._create_value_preview(value, max_items=100)

        # Handle collections
        elif isinstance(value, (list, tuple, dict)):
            preview = self._create_value_preview(value, max_items=10)

        # Handle unknown types
        else:
            preview = ValuePreview(text=f"<{type_name} object>", data=None)

        # Generate unique ID for this variable record
        var_id = f"var:{runtime_id}:{slot_path}"

        return VariableRecord(
            id=var_id,
            slot_path=slot_path,
            display_name=name,
            runtime_object_id=runtime_id,
            type_name=type_name,
            tensor_meta=tensor_meta,
            tensor_stats=tensor_stats,
            preview=preview,
            producer_call_id=None,  # Will be set by dataflow resolver
            consumer_call_ids=[],  # Will be populated by dataflow resolver
        )

    def _extract_tensor_meta(self, tensor: Any) -> TensorMeta:
        """Extract metadata from a torch tensor.

        Args:
            tensor: Torch tensor object

        Returns:
            TensorMeta containing shape, dtype, device, etc.
        """
        try:
            shape = list(tensor.shape) if hasattr(tensor, "shape") else None
            dtype = str(tensor.dtype) if hasattr(tensor, "dtype") else None
            device = str(tensor.device) if hasattr(tensor, "device") else None
            requires_grad = (
                tensor.requires_grad if hasattr(tensor, "requires_grad") else None
            )
            numel = tensor.numel() if hasattr(tensor, "numel") else None

            return TensorMeta(
                shape=shape,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
                numel=numel,
            )
        except Exception:
            # If extraction fails, return empty metadata
            return TensorMeta(
                shape=None,
                dtype=None,
                device=None,
                requires_grad=None,
                numel=None,
            )

    def _compute_tensor_stats(
        self, tensor: Any, policy: TensorValuePolicy
    ) -> TensorStats | None:
        """Compute statistical summary of tensor values.

        Args:
            tensor: Torch tensor object
            policy: Policy determining what statistics to compute

        Returns:
            TensorStats if policy allows, None otherwise
        """
        if policy == "none":
            return None

        # For all other policies, compute stats
        try:
            # Only compute stats for floating point tensors
            if not hasattr(tensor, "dtype"):
                return None

            dtype_str = str(tensor.dtype)
            if "float" not in dtype_str and "double" not in dtype_str:
                return None

            # Compute statistics
            min_val = float(tensor.min().item()) if tensor.numel() > 0 else None
            max_val = float(tensor.max().item()) if tensor.numel() > 0 else None
            mean_val = float(tensor.mean().item()) if tensor.numel() > 0 else None
            std_val = float(tensor.std().item()) if tensor.numel() > 0 else None

            return TensorStats(
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        except Exception:
            # If computation fails, return None
            return None

    def _create_tensor_preview(
        self, tensor: Any, policy: TensorValuePolicy
    ) -> ValuePreview:
        """Create preview representation of tensor values.

        Args:
            tensor: Torch tensor object
            policy: Policy determining what preview to create

        Returns:
            ValuePreview with text and/or data representation
        """
        try:
            numel = tensor.numel() if hasattr(tensor, "numel") else 0
            shape = tuple(tensor.shape) if hasattr(tensor, "shape") else ()

            # Create text representation
            text = f"Tensor{shape}"

            # Determine if we should include values
            data = None
            if policy == "full":
                # Include all values (be careful with large tensors!)
                if numel <= 4096:
                    data = tensor.detach().cpu().tolist()
                else:
                    # Even for "full" policy, truncate very large tensors
                    data = tensor.flatten()[:4096].detach().cpu().tolist()
                    text += " (truncated)"
            elif policy == "preview":
                # Include first N values
                if numel <= 32:
                    data = tensor.detach().cpu().tolist()
                else:
                    data = tensor.flatten()[:32].detach().cpu().tolist()
                    text += f" (showing first 32 of {numel})"
            # For "stats_only" and "none", data remains None

            return ValuePreview(text=text, data=data)
        except Exception:
            # If preview creation fails, return minimal info
            return ValuePreview(text="<Tensor>", data=None)

    def _create_value_preview(self, value: Any, max_items: int = 10) -> ValuePreview:
        """Create preview representation for non-tensor values.

        Args:
            value: Value to preview
            max_items: Maximum number of items to include for collections

        Returns:
            ValuePreview with text and/or data representation
        """
        try:
            # Handle None
            if value is None:
                return ValuePreview(text="None", data=None)

            # Handle primitives
            if isinstance(value, bool):
                return ValuePreview(text=str(value), data=value)
            elif isinstance(value, (int, float)):
                return ValuePreview(text=str(value), data=value)
            elif isinstance(value, str):
                # Truncate long strings
                if len(value) <= 100:
                    return ValuePreview(text=repr(value), data=value)
                else:
                    truncated = value[:100] + "..."
                    return ValuePreview(text=repr(truncated), data=truncated)

            # Handle lists
            elif isinstance(value, list):
                if len(value) == 0:
                    return ValuePreview(text="[]", data=[])
                elif len(value) <= max_items:
                    text = f"list[{len(value)}]"
                    return ValuePreview(text=text, data=value[:max_items])
                else:
                    text = f"list[{len(value)}] (showing first {max_items})"
                    return ValuePreview(text=text, data=value[:max_items])

            # Handle tuples
            elif isinstance(value, tuple):
                if len(value) == 0:
                    return ValuePreview(text="()", data=())
                elif len(value) <= max_items:
                    text = f"tuple[{len(value)}]"
                    return ValuePreview(text=text, data=value[:max_items])
                else:
                    text = f"tuple[{len(value)}] (showing first {max_items})"
                    return ValuePreview(text=text, data=value[:max_items])

            # Handle dicts
            elif isinstance(value, dict):
                if len(value) == 0:
                    return ValuePreview(text="{}", data={})
                elif len(value) <= max_items:
                    text = f"dict[{len(value)}]"
                    return ValuePreview(
                        text=text, data=dict(list(value.items())[:max_items])
                    )
                else:
                    text = f"dict[{len(value)}] (showing first {max_items})"
                    return ValuePreview(
                        text=text, data=dict(list(value.items())[:max_items])
                    )

            # Handle other types
            else:
                type_name = type(value).__qualname__
                return ValuePreview(text=f"<{type_name}>", data=None)

        except Exception:
            # If preview creation fails, return minimal info
            return ValuePreview(text="<value>", data=None)

    def _standardize_value(self, value: Any) -> Any | dict[str, Any]:
        """Standardize return values into slot path structure.

        Args:
            value: Return value to standardize

        Returns:
            Original value if simple, dict of slot_path -> value if structured
        """
        # Handle None
        if value is None:
            return value

        # Handle tuple returns
        if isinstance(value, tuple):
            result = {}
            for i, item in enumerate(value):
                result[f"return.{i}"] = item
            return result

        # Handle list returns (treat similar to tuple)
        if isinstance(value, list):
            result = {}
            for i, item in enumerate(value):
                result[f"return.{i}"] = item
            return result

        # Handle dict returns
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                # Use string representation of key for slot path
                key_str = str(key) if not isinstance(key, str) else key
                result[f"return.{key_str}"] = item
            return result

        # For all other types, return as-is (single value)
        return value
