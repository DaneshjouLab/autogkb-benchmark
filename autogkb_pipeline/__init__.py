"""
DEPRECATED: autogkb_pipeline has been split into separate packages.

Use:
  - generation.*   (pipeline, modules)
  - benchmark.*    (v1, v2)
  - shared.*       (utils, data_setup, term_normalization)
"""

import warnings

warnings.warn(
    "autogkb_pipeline is deprecated. Import from generation, benchmark, or shared instead.",
    DeprecationWarning,
    stacklevel=2,
)
