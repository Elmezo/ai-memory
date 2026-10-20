"""Alias module for :mod:`policy`."""

from policy import *  # noqa: F401,F403
from policy import (
    MemoryPolicy,
    RuleBasedMemoryPolicy,
    get_memory_policy,
)

__all__ = ["MemoryPolicy", "RuleBasedMemoryPolicy", "get_memory_policy"]
