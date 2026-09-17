#!/usr/bin/env python3
"""
dataset.py

Deprecated location. The tf.data pipeline now lives in build_dataset.py
as create_dataset(), alongside the PGN -> TFRecord CLI that produces the
shards it reads (so the whole "how do I get a tf.data.Dataset out of my
TFRecords" story is in one file). This module just re-exports it so any
existing `from dataset import make_dataset` keeps working.

Prefer importing directly from build_dataset.py in new code:
    from build_dataset import create_dataset
"""

from build_dataset import create_dataset

# Backward-compatible alias for the old name.
make_dataset = create_dataset

__all__ = ["create_dataset", "make_dataset"]
