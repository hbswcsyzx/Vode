"""Trace runtime for capturing function-level execution traces.

This module implements the core tracing mechanism using sys.settrace() to capture
function call events and build a call tree structure.
"""

import sys
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from vode.trace.models import (
    FunctionCallNode,
    TraceGraph,
    TensorValuePolicy,
    VariableRecord,
)
from vode.trace.value_extractor import ValueExtractor
from vode.trace.dataflow_resolver import DataflowResolver


@dataclass
class TraceConfig:
    """Configuration for trace runtime behavior.

    Attributes:
        max_depth: Maximum call stack depth to trace (None = unlimited)
        exclude_patterns: List of regex patterns for files to exclude from tracing
        value_policy: Policy for capturing tensor values (not used in this phase)
        capture_locals: Whether to capture local variables (not used in this phase)
        torch_module_hooks: Whether to install PyTorch module hooks
    """

    max_depth: int | None = None
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            r".*site-packages.*",
            r".*dist-packages.*",
            r".*<frozen.*",
        ]
    )
    value_policy: TensorValuePolicy = "stats_only"
    capture_locals: bool = False
    torch_module_hooks: bool = True


class TraceRuntime:
    """Runtime system for tracing function calls using sys.settrace().

    This class captures function call events and builds a call tree structure.
    It does NOT extract parameter/return values or build dataflow edges - those
    are handled in subsequent phases.
    """

    def __init__(self, config: TraceConfig):
        """Initialize the trace runtime.

        Args:
            config: Configuration for trace behavior
        """
        self.config = config

        # Internal state
        self.call_stack: list[str] = []  # Stack of active call IDs
        self.nodes: dict[str, FunctionCallNode] = {}  # Map call_id -> node
        self.current_depth: int = 0
        self.node_id_counter: int = 0

        # Compiled exclude patterns for performance
        self._exclude_regexes = [
            re.compile(pattern) for pattern in config.exclude_patterns
        ]

        # Track if we're currently tracing
        self._is_tracing: bool = False

        # Optional torch support
        self._torch_available: bool = False
        self._torch_nn_module_class: Any = None
        self._try_import_torch()

        # Value extractor for capturing parameters and return values
        self.value_extractor = ValueExtractor()

        # Storage for all variables captured during tracing
        self.variables: list[VariableRecord] = []

    def _try_import_torch(self) -> None:
        """Attempt to import torch and cache nn.Module class."""
        try:
            import torch.nn as nn

            self._torch_available = True
            self._torch_nn_module_class = nn.Module
        except ImportError:
            self._torch_available = False

    def start(self) -> None:
        """Install the trace function and begin tracing."""
        if self._is_tracing:
            raise RuntimeError("Trace is already active")

        self._is_tracing = True
        sys.settrace(self._trace_function)

        # Install torch hooks if requested and available
        if self.config.torch_module_hooks and self._torch_available:
            self._install_torch_hooks()

    def stop(self) -> TraceGraph:
        """Remove the trace function and return the captured trace graph.

        Returns:
            TraceGraph containing all captured function calls
        """
        if not self._is_tracing:
            raise RuntimeError("Trace is not active")

        sys.settrace(None)
        self._is_tracing = False

        # Build the trace graph
        # Root calls are those with no parent (parent_id is None)
        root_call_ids = [
            node.id for node in self.nodes.values() if node.parent_id is None
        ]

        # Create initial graph with captured data
        graph = TraceGraph(
            root_call_ids=root_call_ids,
            function_calls=list(self.nodes.values()),
            variables=self.variables,
            edges=[],  # Will be built by DataflowResolver
        )

        # Resolve dataflow edges
        resolver = DataflowResolver(graph)
        edges = resolver.resolve()
        graph.edges = edges

        return graph

    def _trace_function(self, frame: Any, event: str, arg: Any) -> Callable | None:
        """Main trace callback invoked by sys.settrace().

        Args:
            frame: Current execution frame
            event: Event type ('call', 'return', 'exception', etc.)
            arg: Event-specific argument

        Returns:
            This function (to continue tracing) or None
        """
        try:
            # Check if we should trace this frame
            if not self._should_trace(frame):
                return None

            if event == "call":
                self._handle_call(frame)
            elif event == "return":
                self._handle_return(frame, arg)
            elif event == "exception":
                self._handle_exception(frame, arg)

            return self._trace_function
        except Exception as e:
            # Gracefully handle errors in trace function to avoid breaking user code
            print(f"Warning: Error in trace function: {e}", file=sys.stderr)
            return self._trace_function

    def _should_trace(self, frame: Any) -> bool:
        """Determine if a frame should be traced based on filtering rules.

        Args:
            frame: Execution frame to check

        Returns:
            True if frame should be traced, False otherwise
        """
        # Check depth limit
        if (
            self.config.max_depth is not None
            and self.current_depth >= self.config.max_depth
        ):
            return False

        # Check file path exclusion patterns
        filename = frame.f_code.co_filename

        # Skip Vode internal modules to avoid tracing the tracer itself
        if "/vode/src/trace/" in filename or "/vode/src/vode/" in filename:
            return False

        # Skip standard library
        if "/lib/python" in filename or "/lib64/python" in filename:
            return False

        for regex in self._exclude_regexes:
            if regex.match(filename):
                return False

        return True

    def _handle_call(self, frame: Any) -> None:
        """Process a function call event.

        Args:
            frame: Execution frame for the call
        """
        # Extract frame information
        frame_info = self._extract_frame_info(frame)

        # Generate unique node ID
        call_id = f"call:{self.node_id_counter}"
        self.node_id_counter += 1

        # Determine parent from call stack
        parent_id = self.call_stack[-1] if self.call_stack else None

        # Extract parameters using ValueExtractor
        # Try to get the function object for signature inspection
        func = None
        try:
            func_name = frame.f_code.co_name
            if func_name in frame.f_globals:
                func = frame.f_globals[func_name]
        except Exception:
            pass

        parameters = self.value_extractor.extract_parameters(frame, func, self.config)

        # Store parameter variables and collect their IDs
        arg_variable_ids = []
        for param_name, var_record in parameters.items():
            self.variables.append(var_record)
            arg_variable_ids.append(var_record.id)

        # Create function call node
        node = FunctionCallNode(
            id=call_id,
            parent_id=parent_id,
            qualified_name=frame_info["qualified_name"],
            display_name=frame_info["display_name"],
            filename=frame_info["filename"],
            lineno=frame_info["lineno"],
            depth=self.current_depth,
            arg_variable_ids=arg_variable_ids,
            return_variable_ids=[],  # Will be filled in _handle_return
            metadata=frame_info["metadata"],
        )

        # Store node
        self.nodes[call_id] = node

        # Update call stack
        self.call_stack.append(call_id)
        self.current_depth += 1

    def _handle_return(self, frame: Any, return_value: Any) -> None:
        """Process a function return event.

        Args:
            frame: Execution frame for the return
            return_value: The value being returned
        """
        # Get the current call_id before popping
        if not self.call_stack:
            return

        call_id = self.call_stack[-1]

        # Extract return value using ValueExtractor
        return_result = self.value_extractor.extract_return_value(
            return_value, self.config
        )

        # Store return variables and collect their IDs
        return_variable_ids = []
        if return_result is not None:
            if isinstance(return_result, dict):
                # Multiple return values (tuple/list/dict)
                for slot_path, var_record in return_result.items():
                    self.variables.append(var_record)
                    return_variable_ids.append(var_record.id)
            else:
                # Single return value
                self.variables.append(return_result)
                return_variable_ids.append(return_result.id)

        # Update the node with return variable IDs
        if call_id in self.nodes:
            self.nodes[call_id].return_variable_ids = return_variable_ids

        # Pop from call stack
        self.call_stack.pop()
        self.current_depth -= 1

    def _handle_exception(self, frame: Any, exc_info: tuple) -> None:
        """Process an exception event.

        Args:
            frame: Execution frame where exception occurred
            exc_info: Exception information tuple (type, value, traceback)
        """
        # For now, just handle it like a return (pop stack)
        # More sophisticated exception tracking can be added later
        if self.call_stack:
            call_id = self.call_stack[-1]
            # Mark the node as having an exception
            if call_id in self.nodes:
                self.nodes[call_id].metadata["exception"] = True
                self.nodes[call_id].metadata["exception_type"] = (
                    exc_info[0].__name__ if exc_info[0] else None
                )

            self.call_stack.pop()
            self.current_depth -= 1

    def _extract_frame_info(self, frame: Any) -> dict[str, Any]:
        """Extract function metadata from a frame object.

        Args:
            frame: Execution frame to extract from

        Returns:
            Dictionary containing extracted metadata
        """
        code = frame.f_code

        # Basic info
        function_name = code.co_name
        filename = code.co_filename
        lineno = frame.f_lineno

        # Try to determine if this is a method and get class info
        qualified_name = function_name
        display_name = function_name
        metadata: dict[str, Any] = {}

        # Check if this is a method by looking for 'self' or 'cls'
        frame_locals = frame.f_locals
        self_obj = frame_locals.get("self")
        cls_obj = frame_locals.get("cls")

        if self_obj is not None:
            # This is likely an instance method
            class_name = type(self_obj).__name__
            module_name = type(self_obj).__module__

            if module_name and module_name != "__main__":
                qualified_name = f"{module_name}.{class_name}.{function_name}"
            else:
                qualified_name = f"{class_name}.{function_name}"

            display_name = f"{class_name}.{function_name}"

            # Check if this is a torch.nn.Module
            if self._torch_available and isinstance(
                self_obj, self._torch_nn_module_class
            ):
                metadata["is_torch_module"] = True
                metadata["module_class"] = class_name

                # Try to get module path if available
                if hasattr(self_obj, "_get_name"):
                    metadata["module_type"] = self_obj._get_name()

        elif cls_obj is not None:
            # This is likely a class method
            class_name = cls_obj.__name__
            module_name = cls_obj.__module__

            if module_name and module_name != "__main__":
                qualified_name = f"{module_name}.{class_name}.{function_name}"
            else:
                qualified_name = f"{class_name}.{function_name}"

            display_name = f"{class_name}.{function_name}"

        else:
            # Regular function - try to get module from globals
            frame_globals = frame.f_globals
            module_name = frame_globals.get("__name__")

            if module_name and module_name != "__main__":
                qualified_name = f"{module_name}.{function_name}"
                display_name = function_name

        return {
            "qualified_name": qualified_name,
            "display_name": display_name,
            "filename": filename,
            "lineno": lineno,
            "metadata": metadata,
        }

    def _install_torch_hooks(self) -> None:
        """Install forward hooks on torch.nn.Module instances.

        This is a minimal implementation for now. More sophisticated hook
        management can be added in later phases.
        """
        # For now, this is a placeholder
        # In a full implementation, we would:
        # 1. Track module instances as they're created
        # 2. Install forward hooks to capture module-level information
        # 3. Correlate hook events with trace events
        #
        # However, since we're using sys.settrace(), we already capture
        # module forward calls, so hooks are optional enhancement
        pass
