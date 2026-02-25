"""Shared pipeline-control primitives."""


class PipelineCancelled(Exception):
    """Raised when a running pipeline is interrupted by user request."""

