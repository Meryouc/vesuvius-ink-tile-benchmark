"""Compare cheap spatial filters on published 2.399 µm ink predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
import tifffile
import zarr


# Half-open crops cover the September 2026 supervision/validation masks
# with at least 32 pixels of context for the spatial filters.
DEV_CROP = (4564, 8170, 1001, 5738)
HOLDOUT_CROPS = {
    "0814": (6466, 8170, 1001, 3297),
    "1667": (12408, 20724, 19637, 28908),
}


def candidates(image, selected=None):
    if selected is None or selected == "raw":
        yield "raw", image
    for sigma in (0.75, 1.5, 3.0, 6.0):
        name = f"gaussian_{sigma:g}"
        if selected is None or selected == name:
            yield name, ndimage.gaussian_filter(image, sigma=sigma, output=np.float32)
    for size in (3, 5):
        name = f"median_{size}"
        if selected is None or selected == name:
            yield name, ndimage.median_filter(image, size=size)
        name = f"opening_{size}"
        if selected is None or selected == name:
            yield name, ndimage.grey_opening(image, size=(size, size))


def score_histogram(scores, truth):
    """Use 256 score levels for all methods, with tied scores handled together."""
    scores = np.rint(scores).astype(np.uint8)
    positives = np.bincount(scores[truth], minlength=256)
    negatives = np.bincount(scores[~truth], minlength=256)
    true_positives = np.cumsum(positives[::-1])
    false_positives = np.cumsum(negatives[::-1])
    positive_count = int(true_positives[-1])
    negative_count = int(false_positives[-1])
    if not positive_count or not negative_count:
        raise ValueError("Evaluation region must contain ink and background.")
    selected_count = true_positives + false_positives
    precision = np.divide(
        true_positives, selected_count, out=np.ones(256), where=selected_count > 0
    )
    recall = true_positives / positive_count
    average_precision = float(np.sum(precision * np.diff(recall, prepend=0)))
    threshold_index = 255 - 128
    tp = int(true_positives[threshold_index])
    fp = int(false_positives[threshold_index])
    return {
        "average_precision": average_precision,
        "dice_at_128": 2 * tp / (tp + fp + positive_count),
        "recall_at_128": tp / positive_count,
        "false_positive_pixels_at_128": fp,
        "positive_pixels": positive_count,
        "negative_pixels": negative_count,
    }


def evaluate(data, case, crop, development=False, selected=None):
    folder = data / case
    y0, y1, x0, x1 = crop
    region = np.s_[y0:y1, x0:x1]
    labels = zarr.open_array(folder / "inklabels", mode="r")
    validation = zarr.open_array(folder / "validation", mode="r")
    supervision = zarr.open_array(folder / "supervision", mode="r")
    with tifffile.TiffFile(folder / "prediction.tif") as tiff, tiff.aszarr() as store:
        prediction = zarr.open_array(store, mode="r")
        if len({array.shape for array in (labels, validation, supervision, prediction)}) != 1:
            raise ValueError(f"Unaligned arrays for {case}")
        image = prediction[region]
    validation_mask = validation[region] != 0
    supervision_mask = supervision[region] != 0
    if not np.all(supervision_mask[validation_mask]):
        raise ValueError(f"Validation contains pixels outside supervision for {case}")
    mask = supervision_mask & ~validation_mask if development else validation_mask
    truth = labels[region][mask] != 0
    methods = [("raw", image)] if selected and selected != "raw" else []
    results = {name: score_histogram(values[mask], truth) for name, values in methods}
    for name, values in candidates(image, selected):
        results[name] = score_histogram(values[mask], truth)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    development = evaluate(args.data, "0814", DEV_CROP, development=True)
    winner = max(development, key=lambda name: development[name]["average_precision"])
    print(f"Development selected {winner}", flush=True)
    holdouts = {}
    for case, crop in HOLDOUT_CROPS.items():
        holdouts[case] = evaluate(args.data, case, crop, selected=winner)
        print(case, json.dumps(holdouts[case]), flush=True)
    report = {"selected": winner, "development_0814": development, "validation": holdouts}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
