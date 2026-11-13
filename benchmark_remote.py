"""Score labelled regions of public Vesuvius TIFFs using HTTP byte ranges."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from hashlib import sha256
import io
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import requests
import tifffile
import zarr

from ink_postprocess import score_histogram


SERVER = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
SOURCES = {
    "0814": ("PHerc0814", "20260226000000-46527_2um_try2", "20260309142202"),
    "1667": ("PHerc1667", "20251212185248-w029_20251212185248662_flatboi", "20251217075048"),
    "0139-044": ("PHerc0139", "20260115000000-w044_2026011522", "20260102150214"),
    "0139-045": ("PHerc0139", "20260126000000-w045_2026012619", "20260102150214"),
    "1667-018": ("PHerc1667", "20240304144031-w018_20240304144031_flatboi", "20251217075048"),
    "1667-023": ("PHerc1667", "20240304161941-w023_20240304161941_flatboi", "20251217075048"),
    "1667-028": ("PHerc1667", "20251208130119-w028_20251208130119156_flatboi", "20251217075048"),
    "1667-031": ("PHerc1667", "20251223230000-w031_2025122323_flatboi", "20251217075048"),
}


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.name = url.rsplit("/", 1)[-1]
        self.position = 0
        self.bytes_read = 0
        self.requests = 0
        self.prefetched = {}
        self.session = requests.Session()
        try:
            response = self.session.head(url, timeout=30)
            response.raise_for_status()
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers["ETag"]
        except Exception:
            self.session.close()
            raise
        self.cache = lru_cache(maxsize=64)(self.fetch)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        position = origin + offset
        if position < 0:
            raise ValueError("Cannot seek before the start of the TIFF.")
        self.position = position
        return position

    def request_range(self, start, end, get):
        with get(
            self.url, headers={"Range": f"bytes={start}-{end - 1}", "If-Match": self.etag},
            stream=True, timeout=30,
        ) as response:
            expected = f"bytes {start}-{end - 1}/{self.size}"
            if (
                response.status_code != 206
                or response.headers.get("Content-Range") != expected
                or response.headers.get("ETag") != self.etag
            ):
                raise OSError(f"Server did not return the requested TIFF range: {expected}")
            content = response.content
        if len(content) != end - start:
            raise OSError("Server returned an incomplete TIFF range.")
        return content

    def fetch(self, start, end):
        content = self.request_range(start, end, self.session.get)
        self.bytes_read += len(content)
        self.requests += 1
        return content

    def prefetch_tiles(self, page, tiles, workers):
        ranges = [
            (page.dataoffsets[tile], page.dataoffsets[tile] + page.databytecounts[tile])
            for tile in tiles
            if page.databytecounts[tile]
        ]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key, content in zip(ranges, pool.map(self.download_tile, ranges), strict=True):
                self.prefetched[key] = content
                self.bytes_read += len(content)
                self.requests += 1

    def download_tile(self, bounds):
        return self.request_range(*bounds, requests.get)

    def read(self, size=-1):
        start = min(self.position, self.size)
        end = self.size if size < 0 else min(start + size, self.size)
        if start == end:
            return b""
        content = self.prefetched.get((start, end))
        if content is None:
            content = self.cache(start, end)
        self.position = end
        return content

    def close(self):
        if hasattr(self, "cache"):
            self.cache.cache_clear()
        self.prefetched.clear()
        self.session.close()
        super().close()


def fetch_label_arrays(folder, label_url):
    files = {}
    for name in ("inklabels", "validation"):
        target = folder / name
        for leaf in ("zarr.json", "c/0/0"):
            path = target / leaf
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                response = requests.get(f"{label_url}/{name}.zarr/0/{leaf}", timeout=30)
                response.raise_for_status()
                temporary = path.with_suffix(".part")
                temporary.write_bytes(response.content)
                temporary.replace(path)
        files[name] = zarr.open_array(target, mode="r")
    return files


def masked_tiles(mask, crop, tile_shape, width):
    y0, _, x0, _ = crop
    tile_height, tile_width = tile_shape
    stride = (width + tile_width - 1) // tile_width
    rows, cols = np.nonzero(mask)
    tile_ids = (rows + y0) // tile_height * stride + (cols + x0) // tile_width
    return rows, cols, np.unique(tile_ids)


def read_masked_tiles(raster, rows, cols, tile_ids, crop):
    y0, _, x0, _ = crop
    tile_height, tile_width = raster.chunks
    stride = (raster.shape[1] + tile_width - 1) // tile_width
    rows += y0
    cols += x0
    point_tiles = rows // tile_height * stride + cols // tile_width
    order = np.argsort(point_tiles, kind="stable")
    counts = np.bincount(point_tiles, minlength=tile_ids[-1] + 1)
    groups = np.split(order, np.cumsum(counts[tile_ids])[:-1])
    scores = np.empty(len(rows), dtype=raster.dtype)
    for tile, group in zip(tile_ids, groups, strict=True):
        tile_row, tile_col = divmod(int(tile), stride)
        top, left = tile_row * tile_height, tile_col * tile_width
        pixels = raster[top:top + tile_height, left:left + tile_width]
        scores[group] = pixels[rows[group] - top, cols[group] - left]
    return scores


def occupied_bounds(mask):
    height, width = mask.shape
    y0, y1, x0, x1 = height, 0, width, 0
    for start in range(0, height, 1024):
        strip = mask[start:start + 1024]
        rows = np.flatnonzero(strip.any(axis=1))
        if rows.size:
            cols = np.flatnonzero(strip.any(axis=0))
            y0 = min(y0, start + int(rows[0]))
            y1 = start + int(rows[-1]) + 1
            x0 = min(x0, int(cols[0]))
            x1 = max(x1, int(cols[-1]) + 1)
    if y0 >= y1:
        raise ValueError("Validation mask has no labelled pixels.")
    return y0, y1, x0, x1


def benchmark(case, data, workers, prediction_url=None):
    scroll, segment, scan = SOURCES[case]
    segment_id = segment.split("-", 1)[0]
    base = f"{SERVER}{scroll}/segments/{segment}"
    prediction = prediction_url or (
        f"{base}/ink-detection/{scroll}-{segment_id}-2.399um-0.22m-78keV-volume-{scan}"
        "-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif"
    )
    label_url = f"{base}/ink-labels/2.399um-volume-{scan}/20260918"
    arrays = fetch_label_arrays(data / case, label_url)
    bounds = occupied_bounds(arrays["validation"])
    y0, y1, x0, x1 = bounds
    crop = np.s_[y0:y1, x0:x1]
    valid = arrays["validation"][crop] != 0
    truth = arrays["inklabels"][crop][valid] != 0
    start = perf_counter()
    with RangeFile(prediction) as remote, tifffile.TiffFile(remote) as tiff, tiff.aszarr() as store:
        raster = zarr.open_array(store, mode="r")
        if raster.shape != arrays["inklabels"].shape:
            raise ValueError(f"Prediction and labels use different grids for {case}")
        page = tiff.pages[0]
        if not page.is_tiled or raster.chunks != (page.tilelength, page.tilewidth):
            raise ValueError(f"Prediction is missing a compatible tiled TIFF layout for {case}")
        rows, cols, tiles = masked_tiles(valid, bounds, raster.chunks, raster.shape[1])
        if workers > 1:
            remote.prefetch_tiles(page, tiles, workers)
        scores = read_masked_tiles(raster, rows, cols, tiles, bounds)
        bytes_read, requests_made, total = remote.bytes_read, remote.requests, remote.size
    return {
        "prediction": prediction,
        "labels": label_url,
        "tiff_bytes_read": bytes_read,
        "tiff_size": total,
        "range_requests": requests_made,
        "tiles_read": len(tiles),
        "tiff_read_seconds": round(perf_counter() - start, 3),
        "masked_score_sha256": sha256(scores.tobytes()).hexdigest(),
        "scores": score_histogram(scores, truth),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--case", action="append", choices=SOURCES)
    parser.add_argument("--prediction-url", help="Score a different tiled TIFF on the same grid")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    if args.prediction_url and (
        len(args.case or []) != 1 or not args.prediction_url.startswith("https://")
    ):
        parser.error("--prediction-url requires one --case and an HTTPS URL")
    results = {}
    for case in args.case or SOURCES:
        results[case] = benchmark(case, args.data, args.workers, args.prediction_url)
        print(case, json.dumps(results[case]), flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
