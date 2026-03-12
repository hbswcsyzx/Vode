"""Function flow tracer for VODE.

Captures Python function call flow using sys.settrace().
Tracks function calls, arguments, return values, and execution time.
"""

import sys
import time
from typing import Any, Callable, Optional
from collections import defaultdict

from vode.core import ComputationGraph, FunctionNode, generate_node_id, sanitize_name


class FunctionTracer:
    """Traces Python function execution flow.
    
    Uses sys.settrace() to capture all function calls in the program,
    building a call graph with timing and argument information.
    
    Attributes:
        capture_data: Whether to capture function arguments and return values
        max_depth: Maximum call stack depth to capture
        filter_patterns: List of module patterns to exclude (e.g., ['torch.', 'numpy.'])
    """
    
    def __init__(
        self,
        capture_data: bool = False,
        max_depth: Optional[int] = None,
        filter_patterns: Optional[list[str]] = None
    ):
        """Initialize function tracer.
        
        Args:
            capture_data: Whether to capture arguments and return values
            max_depth: Maximum call depth to trace (None for unlimited)
            filter_patterns: Module patterns to exclude from tracing
        """
        self.capture_data = capture_data
        self.max_depth = max_depth
        self.filter_patterns = filter_patterns or ['<frozen', 'importlib', '_bootstrap']
        
        self.graph = ComputationGraph()
        self._call_stack: list[str] = []
        self._node_counter = 0
        self._function_times: dict[str, float] = {}
        self._function_calls: defaultdict[str, int] = defaultdict(int)
    
    def _should_trace(self, frame) -> bool:
        """Check if a frame should be traced.
        
        Args:
            frame: Python frame object
            
        Returns:
            True if frame should be traced
        """
        # Check depth limit
        if self.max_depth and len(self._call_stack) >= self.max_depth:
            return False
        
        # Check filter patterns
        code = frame.f_code
        filename = code.co_filename
        
        for pattern in self.filter_patterns:
            if pattern in filename:
                return False
        
        return True
    
    def _trace_function(self, frame, event: str, arg: Any):
        """Trace function callback for sys.settrace().
        
        Args:
            frame: Current frame
            event: Event type ('call', 'return', 'line', etc.)
            arg: Event argument
        """
        if event == 'call':
            if not self._should_trace(frame):
                return None
            
            code = frame.f_code
            func_name = code.co_filename
            
            # Create function node
            node_id = generate_node_id('func')
            node = FunctionNode(
                node_id=node_id,
                name=sanitize_name(func_name),
                depth=len(self._call_stack),
                func_name=func_name,
                qualified_name=f"{code.co_filename}:{code.co_name}",
            )
            
            # Capture arguments if requested
            if self.capture_data:
                # TODO: Capture function arguments from frame.f_locals
                pass
            
            self.graph.add_node(node)
            
            # Track call stack
            if self._call_stack:
                parent_id = self._call_stack[-1]
                self.graph.add_edge(parent_id, node_id)
            
            self._call_stack.append(node_id)
            self._function_times[node_id] = time.time()
            self._function_calls[func_name] += 1
            
            return self._trace_function
        
        elif event == 'return':
            if self._call_stack:
                node_id = self._call_stack.pop()
                
                # Record execution time
                if node_id in self._function_times:
                    elapsed = time.time() - self._function_times[node_id]
                    node = self.graph.get_node(node_id)
                    if node:
                        node.metadata['execution_time'] = elapsed
                
                # Capture return value if requested
                if self.capture_data and arg is not None:
                    # TODO: Capture return value
                    pass
        
        return self._trace_function
    
    def start(self):
        """Start tracing function calls."""
        sys.settrace(self._trace_function)
    
    def stop(self):
        """Stop tracing function calls."""
        sys.settrace(None)
    
    def capture(self, target: Callable, *args: Any, **kwargs: Any) -> ComputationGraph:
        """Capture function execution flow.
        
        Args:
            target: Function to trace
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            ComputationGraph with function call flow
        """
        # Reset state
        self.graph = ComputationGraph()
        self._call_stack.clear()
        self._node_counter = 0
        self._function_times.clear()
        self._function_calls.clear()
        
        # Start tracing
        self.start()
        
        try:
            # Execute target function
            result = target(*args, **kwargs)
        finally:
            # Stop tracing
            self.stop()
        
        return self.graph


def capture_function_flow(
    target: Callable,
    *args: Any,
    capture_data: bool = False,
    max_depth: Optional[int] = None,
    **kwargs: Any
) -> ComputationGraph:
    """Capture Python function call flow.
    
    Main API function for function flow capture. Traces function calls
    using sys.settrace() to build a call graph.
    
    Args:
        target: Function to trace
        *args: Arguments to pass to function
        capture_data: Whether to capture arguments and return values
        max_depth: Maximum call depth to trace
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        ComputationGraph with function call flow
        
    Example:
        >>> def my_function(x):
        ...     return x * 2
        >>> graph = capture_function_flow(my_function, 5)
        >>> print(f"Captured {len(graph.nodes)} function calls")
    """
    tracer = FunctionTracer(capture_data=capture_data, max_depth=max_depth)
    return tracer.capture(target, *args, **kwargs)
