"""auxiliary_tasks must fail at startup on datasets that cannot generate them (#1490).

ZarrDataset filtered auxiliary targets out of target_names and derived nothing
for them, so the aux heads trained against zeros for 50 epochs with
``Avg Loss = 0.0000`` and no warning. Both datasets now raise before any I/O.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vesuvius.models.datasets.zarr_dataset import ZarrDataset, _reject_auxiliary_targets


def _mgr(targets):
    return SimpleNamespace(
        data_path="/nonexistent/data",
        train_patch_size=[64, 64, 64],
        targets=targets,
        profile_augmentations=False,
    )


def test_zarr_dataset_rejects_auxiliary_targets_before_io():
    targets = {
        "surface": {"activation": "none", "out_channels": 1},
        "distance_transform": {"auxiliary_task": True, "source_target": "surface"},
    }
    with pytest.raises(ValueError, match=r"auxiliary_tasks \['distance_transform'\].*#1490"):
        ZarrDataset(_mgr(targets), is_training=True)


def test_helper_ignores_regular_targets():
    _reject_auxiliary_targets({"ink": {"activation": "sigmoid"}}, "ZarrDataset")
    _reject_auxiliary_targets({}, "ZarrDataset")
    _reject_auxiliary_targets(None, "ZarrDataset")
