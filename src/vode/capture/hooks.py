"""PyTorch hooks utilities for VODE capture.

Provides helper functions for registering and managing PyTorch module hooks.
"""

from typing import Any, Callable
import torch.nn as nn


def register_forward_hooks(
    module: nn.Module,
    pre_hook: Callable | None = None,
    post_hook: Callable | None = None,
) -> list[Any]:
    """Register forward hooks on all modules in a model.

    Args:
        module: Root module to register hooks on
        pre_hook: Pre-forward hook function (called before forward)
        post_hook: Post-forward hook function (called after forward)

    Returns:
        List of hook handles that can be used to remove hooks
    """
    hooks = []

    for mod in module.modules():
        if pre_hook:
            handle = mod.register_forward_pre_hook(pre_hook)
            hooks.append(handle)

        if post_hook:
            handle = mod.register_forward_hook(post_hook)
            hooks.append(handle)

    return hooks


def remove_hooks(hooks: list[Any]) -> None:
    """Remove all hooks from a list of hook handles.

    Args:
        hooks: List of hook handles to remove
    """
    for hook in hooks:
        hook.remove()
