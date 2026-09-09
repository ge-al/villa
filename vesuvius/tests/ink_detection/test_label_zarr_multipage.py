"""Multi-page label TIFFs must be rejected, not silently mangled (#1738)."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from vesuvius.ink_detection.preprocessing.create_label_zarrs import (
    _normalize_to_2d,
    _normalized_2d_shape,
    load_image,
)


def test_multipage_tiff_is_rejected_with_page_count(tmp_path):
    path = tmp_path / "labels_multipage.tif"
    tifffile.imwrite(path, np.zeros((5, 40, 60), dtype=np.uint8))  # 5 pages of 40x60
    with pytest.raises(ValueError, match=r"multi-page TIFF \(5 pages\)"):
        load_image(path)


def test_single_page_rgb_still_loads_as_2d(tmp_path):
    path = tmp_path / "labels_rgb.tif"
    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    rgb[..., 0] = 7
    tifffile.imwrite(path, rgb, photometric="rgb")
    image = load_image(path)
    assert image.shape == (40, 60) and int(image[0, 0]) == 7


def test_page_stacked_array_is_not_mistaken_for_channels(tmp_path):
    with pytest.raises(ValueError, match="Expected a 2D image"):
        _normalize_to_2d(np.zeros((5, 40, 60), dtype=np.uint8), tmp_path / "x.tif")
    with pytest.raises(ValueError, match="Expected a 2D image"):
        _normalized_2d_shape((5, 40, 60), tmp_path / "x.tif")
    assert _normalized_2d_shape((40, 60, 3), tmp_path / "x.tif") == (40, 60)
