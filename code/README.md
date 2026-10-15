# HAPACT Code

This code is the official implementation of the paper **HAPACT: A Benchmark For Human-Centric Physical Impact Localization in Movies** (NeurIPS 2026).

<!-- TODO: Link the paper title to its NeurIPS proceedings page when available. -->
<!-- [paper]: TODO_NEURIPS_PROCEEDINGS_URL -->

[![Project Page](https://img.shields.io/badge/Project-Page-326EA8?style=flat-square)](https://hci-tech-lab.github.io/HAPACT/) ![arXiv: Coming Soon](https://img.shields.io/badge/arXiv-Coming%20Soon-B31B1B?style=flat-square) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Dataset-F2B544?style=flat-square)](https://huggingface.co/datasets/HAPACT/HAPACT-Dataset)

<!-- TODO: Replace the arXiv placeholder with a linked badge once the preprint is available. -->

This directory provides code for preparing the HAPACT dataset and source videos, along with code for the temporal and spatial benchmarks and for training and evaluating the MLLM baseline model. For an overview of the project and paper, see the [project page](https://hci-tech-lab.github.io/HAPACT/).

All commands below should be run from the **`HAPACT/code/`** directory.

## ⚙️ Installation and Setup

### 1. Clone the Repository and Create an Environment

```bash
git clone https://github.com/HCI-Tech-Lab/HAPACT.git
cd HAPACT/code

conda create -n hapact python=3.11 -y
conda activate hapact
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

We tested the code with Python 3.9 and CUDA 11.8. For other platforms or CUDA versions, refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```


## 📦 Preparing the HAPACT Dataset

The steps below download the original annotations, convert their frame indices to the OpenCV-compatible format, and optionally generate benchmark files and source media. Each script processes the **train**, **val**, and **test** splits.

### 1. Download Annotations

Download the original HAPACT annotation files from [Hugging Face](https://huggingface.co/datasets/HAPACT/HAPACT-Dataset):

```bash
# Annotation files
hf download HAPACT/HAPACT-Dataset hapact_train.json hapact_val.json hapact_test.json --repo-type dataset --local-dir ./data
```

The downloaded annotations are saved as `data/hapact_train.json`, `data/hapact_val.json`, and `data/hapact_test.json`.

### 2. Prepare Annotations and Benchmark Files

Run `prepare_data.py` for the desired output under `data/`:

**Annotations only** — generates OpenCV-compatible annotations in `data/opencv/`:

```bash
python prepare_data.py
```

**Temporal localization benchmark** — additionally generates `data/temporal/train.jsonl`, `data/temporal/val.jsonl`, and `data/temporal/test.jsonl`:

```bash
python prepare_data.py --temporal
```

**Spatial localization benchmark** — additionally generates `data/spatial/train.jsonl`, `data/spatial/val.jsonl`, and `data/spatial/test.jsonl`:

```bash
python prepare_data.py --spatial

# (Optional) Required for spatial vertex-level evaluation
hf download HAPACT/HAPACT smplx_neutral_geodesic_dist.npy --repo-type dataset --local-dir ./data
```

To use a custom body-part segmentation, modify `data/smplx_part_segmentation.json` by specifying the SMPL-X vertex indices corresponding to each body region before running with `--spatial`.

### 3. Download Videos and Extract Frames

Video clips are downloaded from their original YouTube sources using [yt-dlp](https://github.com/yt-dlp/yt-dlp) and encoded at 24 FPS. [FFmpeg](https://ffmpeg.org/download.html) (including `ffprobe`) must be installed and available on your `PATH`. [Deno](https://docs.deno.com/runtime/getting_started/installation/) is also required for YouTube JavaScript challenge handling in recent yt-dlp versions.

**Download video clips** — saves MP4 files to `data/source_clips/`:

```bash
python download_file.py
```

**Download video clips and extract spatial frames** — additionally extracts frames required for the spatial benchmark into `data/images/`:

```bash
python download_file.py --spatial
```

For videos that require a browser session, the optional `--cookies-from-browser` argument can be used, for example:

```bash
python download_file.py --spatial --cookies-from-browser chrome
```

> **Video availability.** Video availability may vary depending on geographic restrictions, copyright restrictions, or changes to the original YouTube sources. Unavailable videos are skipped during download. Re-running the script retries unavailable videos while skipping previously downloaded valid clips. The annotation and benchmark JSONL files are not modified when a source video is unavailable; some referenced image files may therefore be missing.


### Resulting Directory Structure

After preparing both benchmarks and downloading spatial frames, the generated data is organized as follows:

```text
code/
└── data/
    ├── hapact_train.json
    ├── hapact_val.json
    ├── hapact_test.json
    ├── opencv_mapping.npz
    ├── smplx_part_segmentation.json
    ├── opencv/
    │   ├── hapact_train_opencv.json
    │   ├── hapact_val_opencv.json
    │   └── hapact_test_opencv.json
    ├── temporal/
    │   ├── train.jsonl
    │   ├── val.jsonl
    │   └── test.jsonl
    ├── spatial/
    │   ├── train.jsonl
    │   ├── val.jsonl
    │   └── test.jsonl
    ├── source_clips/
    │   └── {segment_id}.mp4
    └── images/
        └── {segment_id}_{frame:06d}.png
```

`data/temporal/`, `data/spatial/`, and `data/images/` are created only when the corresponding options are used. Spatial JSONL entries reference individual impact frames using `imgname` values such as `./{segment_id}_{frame:06d}.png`, relative to the `data/images/` directory.

## 📊 Benchmark Evaluation

### Temporal Localization

Generate predictions using your own model and save them as `data/temporal/prediction.jsonl`, with one entry per query. Each entry should contain the `qid` from the corresponding benchmark split and the predicted temporal windows in seconds:

```json
{"qid":"example_segment_id","pred_relevant_windows":[[1.0,1.25,0.95],[2.5,2.75,0.80]]}
```

Each window can be specified as `[start, end]` or `[start, end, score]`. If the confidence score is omitted, it defaults to `1.0`.

Run the evaluation for the desired split:

```bash
python eval_temporal.py --split test
```

The script evaluates `data/temporal/prediction.jsonl` against `data/temporal/test.jsonl` and saves the metrics to `result/temporal/test_metrics.json`.

### Spatial Localization

Generate predictions using your own model and save them as `data/spatial/prediction.jsonl`, with one entry per annotated frame and impact query. Each entry should contain the `segment_id`, `sentence_id`, and `target_frame` from the corresponding benchmark split, along with the predicted body-region labels or SMPL-X vertex indices:

**Region-label predictions** — use region names defined in `data/smplx_part_segmentation.json`:

```json
{"segment_id":"example_segment_id","sentence_id":"0","target_frame":120,"pred_regions":["left forearm","right hand"]}
```

**Vertex predictions** — use SMPL-X vertex indices (0–10474):

```json
{"segment_id":"example_segment_id","sentence_id":"0","target_frame":120,"pred_vertices":[105,106,107,108]}
```

Run the evaluation for the desired mode:

```bash
python eval_spatial.py --split test --region-label
python eval_spatial.py --split test --vertex
```

The script evaluates `data/spatial/prediction.jsonl` against the corresponding benchmark annotations and saves the metrics to `result/spatial/test_region_label_metrics.json` or `result/spatial/test_vertex_metrics.json`, respectively.


## 🤖 MLLM Baseline

We provide a LoRA fine-tuning and evaluation pipeline for [Cosmos-Reason2](https://huggingface.co/nvidia/Cosmos-Reason2-8B), which predicts the body regions affected by an impact using the spatial benchmark.


### Training

Train Cosmos-Reason2-8B using the spatial benchmark and evaluate the saved checkpoints on the validation split (At least 96 GB of GPU memory is recommended for training):

```bash
python cosmos/train.py --run_region_eval \
    --output_dir result/cosmos/query_guided_8b
```


The best checkpoint is saved to `<OUTPUT_DIR>/best_region_checkpoint/`, along with its predictions (`prediction.jsonl`) and metrics (`best_region_metrics.json`) in `<OUTPUT_DIR>`.

### Evaluation

Evaluate a trained checkpoint on the validation or test split using `--resume` to specify the checkpoint directory. At least 32 GB of GPU memory is required for inference.

**Validation:**

```bash
python cosmos/eval.py --split val \
    --resume result/cosmos/query_guided_8b/best_region_checkpoint \
    --save_prediction_path result/cosmos/query_guided_8b/val_prediction.jsonl \
    --save_metrics_path result/cosmos/query_guided_8b/val_metrics.json
```

**Test:**

```bash
python cosmos/eval.py --split test \
    --resume result/cosmos/query_guided_8b/best_region_checkpoint \
    --save_prediction_path result/cosmos/query_guided_8b/test_prediction.jsonl \
    --save_metrics_path result/cosmos/query_guided_8b/test_metrics.json
```

To evaluate a different checkpoint, specify its directory, such as `result/cosmos/query_guided_8b/checkpoint-400`.


### Inference

Run inference on a single image using a trained checkpoint by providing an image and impact query:

```bash
python cosmos/inference.py \
    --image data/images/tt0974015_f9Od8yx9gmg_128.5_135.5_015_000137.png \
    --query "A warrior with a bow slides across the grass on her body." \
    --resume result/cosmos/query_guided_8b/best_region_checkpoint
```

The script prints the raw model output and predicted body-region labels. Use `--resume` to specify a different checkpoint.

## Acknowledgements

The temporal evaluation code is adapted from [QV-M²](https://github.com/Zhuo-Cao/QV-M2), with utilities based on [Moment-DETR](https://github.com/jayleicn/moment_detr) and [MMAction2](https://github.com/open-mmlab/mmaction2). We thank the authors for making their code publicly available.

The source movie clips are obtained from *[Movieclips](https://www.youtube.com/@MOVIECLIPS)* YouTube Channel and remain the property of their respective copyright holders. The videos are not redistributed in this repository.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

This project is licensed under the [MIT License](../LICENSE).