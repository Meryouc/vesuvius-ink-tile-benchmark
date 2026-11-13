# Score Vesuvius ink predictions with fewer downloads

This CPU-only tool scores published tiled ink predictions against the September 2026 validation masks. It reads TIFF metadata and fetches only tiles that contain labelled pixels. It keeps the original pixel order and requires the server to return the correct range from the same object version. It caches the much smaller annotation files locally.

## Reproduce

Tested with Python 3.13 on Windows. From this directory, install the requirements and run the benchmark. Use `--case 0814` for a quick check, or omit it for all eight segments.

```sh
python -m pip install -r requirements.txt
python benchmark_remote.py --data remote-cache --output remote-results.json
```

The JSON records the original source URLs, tile counts, TIFF response bytes, elapsed TIFF read time, pixel counts and scores. TIFF byte counts exclude HTTP headers and annotation files. The output includes average precision and Dice at the fixed threshold of 128 on the published 8-bit prediction. These scores measure agreement with published validation annotations, whose provenance is described by the organizers. [Ink detection tutorial](https://scrollprize.org/tutorial5) · [Data formats and citation guidance](https://scrollprize.org/data_segments)

## Measured on public scroll data

| Segment | Full prediction (MB) | Fetched TIFF data (MB) | Average precision |
|---|---:|---:|---:|
| PHerc. 0814, 46527 | 9.03 | 0.75 | 0.8986 |
| PHerc. 1667, w029 | 120.70 | 1.84 | 0.8293 |
| PHerc. 0139, w044 | 75.93 | 1.17 | 0.6829 |
| PHerc. 0139, w045 | 69.66 | 0.48 | 0.8260 |
| PHerc. 1667, w018 | 340.20 | 4.78 | 0.8589 |
| PHerc. 1667, w023 | 386.18 | 1.91 | 0.7970 |
| PHerc. 1667, w028 | 122.07 | 1.88 | 0.8389 |
| PHerc. 1667, w031 | 88.83 | 3.97 | 0.7226 |

Across these eight predictions the tool transferred **16.78 MB of 1,212.60 MB**, a **98.62%** reduction in TIFF payload. On PHerc. 0814 and PHerc. 1667 w029, the scores of all 8,658,469 labelled pixels matched the local full TIFFs byte for byte. The w029 rectangular crop needed 90 tiles. The actual mask needed 14.

Use this to screen candidate prediction maps against several annotated regions on a low-bandwidth connection. To compare a different prediction, use one `--case` and pass the HTTPS URL of its tiled TIFF to `--prediction-url`. The server must support byte ranges and provide an ETag. The tool checks shape agreement. Verify that the alternate TIFF uses the same alignment and pixel scale as the annotation grid before comparing scores. This experiment evaluates existing predictions and uses annotation agreement as its metric.

## Spatial filtering and scan blend

The filtering trial selected a six-pixel Gaussian from eight filters plus the unmodified baseline on the PHerc. 0814 supervision region outside validation. Average precision increased by approximately 0.0004 on each of two validation regions. At threshold 128, false-positive pixels increased by 286 on PHerc. 0814 and decreased by 728 on PHerc. 1667 w029. Dice moved by less than 0.0002 on each.

A linear blend of the 2.399 µm and 1.129 µm predictions under a simple raster resize did worse on PHerc. 0814. The selected 5% fine-scan blend reached 0.8927 average precision on validation versus 0.8986 for the 2.399 µm prediction alone. The PHerc. 1667 grids had unequal scale ratios and were excluded from this blend experiment. It would take verified cross-scan registration to evaluate that case.

Data used for the PHerc. 1667 tests are from the EduceLab-Scrolls dataset. [Parsons et al.](https://doi.org/10.48550/arXiv.2304.02084) · [Vesuvius data license](https://dl.ash2txt.org/LICENSE.txt)
