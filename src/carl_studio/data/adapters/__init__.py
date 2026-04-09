"""Dataset adapters — transform source-specific formats into UnifiedSample.

Each adapter inherits from DataAdapter and implements adapt().
New sources are added by:
  1. Writing an adapter class
  2. Adding a YAML entry in data_sources.yaml
"""

from carl_studio.data.adapters.base import DataAdapter

__all__ = ["DataAdapter"]
