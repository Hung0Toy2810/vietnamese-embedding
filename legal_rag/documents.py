"""Document metadata and source files are managed by Repository.

This module preserves a discoverable import path for callers migrating from the
old page-based document manager. New code should import Repository directly.
"""

from legal_rag.storage import Repository

__all__ = ["Repository"]
