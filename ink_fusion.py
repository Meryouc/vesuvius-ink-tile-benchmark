"""Measure agreement and fusion of two published scans of the same surface."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
import tifffile
import zarr

from ink_postprocess import DEV_CROP, HOLDOUT_CROPS, score_histogram


class UnverifiedAlignment(ValueError):
    """The rasters cannot be paired by the tested uniform resize."""


def read_pair(folder, crop):
    y0, y1, x0, x1 = crop
    with (
        tifffile.TiffFile(folder / "prediction.tif") as coarse_tiff,
        coarse_tiff.aszarr() as coarse_store,
        tifffile.TiffFile(folder / "prediction_1um.tif") as fine_tiff,
        fine_tiff.aszarr() as fine_store,
    ):
        coarse = zarr.open_array(coarse_store, mode="r")
        fine = zarr.open_array(fine_store, mode="r")
        ratios = np.asarray(fine.shape) / coarse.shape
        if np.ptp(ratios) > 0.002:
            raise UnverifiedAlignment(f"Unequal resize factors for {folder.name}: {ratios}")
        # Both rasters were rendered from the same named segment. Approximate
        # their correspondence by mapping centers between the raster extents.
        rows = (np.arange(y0, y1) + 0.5) * ratios[0] - 0.5
        cols = (np.arange(x0, x1) + 0.5) * ratios[1] - 0.5
        top, left = int(np.floor(rows[0])), int(np.floor(cols[0]))
        bottom, right = int(np.ceil(rows[-1])) + 1, int(np.ceil(cols[-1])) + 1
        fine_roi = fine[top:bottom, left:right]
        grid = np.meshgrid(rows - top, cols - left, indexing="ij")
        fine_resized = ndimage.map_coordinates(
            fine_roi, grid, order=1, mode="nearest", output=np.float32
        )
        return coarse[y0:y1, x0:x1], fine_resized


def score_pair(coarse, fine, truth, methods):
    results = {}
    for name in methods:
        if name == "coarse":
            scores = coarse
        elif name == "fine":
            scores = fine
        else:
            _, weight_text = name.split("_", 1)
            weight = float(weight_text)
            scores = (1 - weight) * coarse + weight * fine
        results[name] = score_histogram(scores, truth)
    return results


def evaluate(folder, crop, methods, development=False):
    y0, y1, x0, x1 = crop
    region = np.s_[y0:y1, x0:x1]
    validation = zarr.open_array(folder / "validation", mode="r")[region] != 0
    supervision = zarr.open_array(folder / "supervision", mode="r")[region] != 0
    mask = supervision & ~validation if development else validation
    truth = zarr.open_array(folder / "inklabels", mode="r")[region][mask] != 0
    coarse, fine = read_pair(folder, crop)
    coarse = coarse[mask].astype(np.float32)
    fine = fine[mask]
    return score_pair(coarse, fine, truth, methods)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    methods = ("coarse", "fine", "blend_0.05", "blend_0.1", "blend_0.25", "blend_0.5", "blend_0.75")
    development = evaluate(args.data / "0814", DEV_CROP, methods, development=True)
    selected = max(development, key=lambda method: development[method]["average_precision"])
    best_blend = max(methods[2:], key=lambda method: development[method]["average_precision"])
    print("Development", json.dumps(development), flush=True)
    validation = {}
    skipped = {}
    for case, crop in HOLDOUT_CROPS.items():
        try:
            validation[case] = evaluate(args.data / case, crop, ("coarse", "fine", best_blend))
        except UnverifiedAlignment as error:
            skipped[case] = str(error)
            print(case, str(error), flush=True)
        else:
            print(case, json.dumps(validation[case]), flush=True)
    if args.output:
        report = {
            "selected": selected,
            "best_blend": best_blend,
            "development_0814": development,
            "validation": validation,
            "skipped": skipped,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
