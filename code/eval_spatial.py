import argparse
import json
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"
NUM_VERTICES = 10475


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_key(item):
    return str(item["segment_id"]), str(item["sentence_id"]), int(item["target_frame"])


def load_predictions(path, gt_keys, field):
    predictions = {}
    for item in load_jsonl(path):
        key = sample_key(item)
        if key in predictions:
            raise ValueError(f"Duplicate prediction: {key}")
        if field not in item:
            raise ValueError(f"Missing '{field}' in prediction: {key}")
        if not isinstance(item[field], list):
            raise ValueError(f"'{field}' must be a list: {key}")
        predictions[key] = item[field]

    missing = gt_keys - predictions.keys()
    extra = predictions.keys() - gt_keys
    if missing or extra:
        raise ValueError(f"Prediction/GT mismatch: {len(missing)} missing, {len(extra)} extra. "
                         f"Missing examples: {sorted(missing)[:5]}; extra examples: {sorted(extra)[:5]}")
    return predictions


def load_regions(path):
    with open(path, "r", encoding="utf-8") as f:
        segmentation = json.load(f)
    return {name: part["vertices"] for name, part in segmentation.items()}


def compute_sample_metrics(pred, gt, eps=1e-8):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    f1 = 2 * precision * recall / max(precision + recall, eps)
    acc = (tp + tn) / max(tp + fp + fn + tn, eps)

    return acc, precision, recall, f1, tp, fp, fn, tn


def compute_vertex_metrics(pred_list, gt_list):
    accs, ps, rs, f1s = [], [], [], []

    total_tp = total_fp = total_fn = total_tn = 0

    for pred, gt in zip(pred_list, gt_list):
        acc, p, r, f1, tp, fp, fn, tn = compute_sample_metrics(pred, gt)
        accs.append(acc)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    micro_p = total_tp / max(total_tp + total_fp, 1e-8)
    micro_r = total_tp / max(total_tp + total_fn, 1e-8)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)
    micro_acc = (total_tp + total_tn) / max(total_tp + total_fp + total_fn + total_tn, 1e-8)

    return {
        "sample_mean_acc": float(np.mean(accs)),
        "sample_mean_precision": float(np.mean(ps)),
        "sample_mean_recall": float(np.mean(rs)),
        "sample_mean_f1": float(np.mean(f1s)),
        "micro_acc": float(micro_acc),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "tn": int(total_tn),
    }


def compute_body_region_vertex_metrics(pred_list, gt_list, region_to_vertices, num_vertices):
    out = {}
    macro_f1s = []

    for region, verts in region_to_vertices.items():
        verts = np.asarray(verts, dtype=np.int64)
        verts = verts[(verts >= 0) & (verts < num_vertices)]

        if len(verts) == 0:
            continue

        total_tp = total_fp = total_fn = total_tn = 0
        sample_f1s = []

        for pred, gt in zip(pred_list, gt_list):
            p = pred[verts]
            g = gt[verts]

            acc, precision, recall, f1, tp, fp, fn, tn = compute_sample_metrics(p, g)
            if np.any(g > 0.5):
                sample_f1s.append(f1)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        micro_p = total_tp / max(total_tp + total_fp, 1e-8)
        micro_r = total_tp / max(total_tp + total_fn, 1e-8)
        micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)
        micro_acc = (total_tp + total_tn) / max(total_tp + total_fp + total_fn + total_tn, 1e-8)

        out[region] = {
            "num_vertices": int(len(verts)),
            "sample_mean_f1": float(np.mean(sample_f1s)),
            "micro_acc": float(micro_acc),
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f1),
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "tn": int(total_tn),
        }

        macro_f1s.append(micro_f1)

    summary = {
        "body_region_vertex_macro_f1": float(np.mean(macro_f1s)) if len(macro_f1s) > 0 else 0.0,
        "body_region_vertex_num_regions": int(len(out)),
    }

    return summary, out


def compute_per_region_label_metrics(pred_region_list, gt_region_list, region_names):
    out = {}
    macro_f1s = []
    macro_ps = []
    macro_rs = []

    for region in region_names:
        total_tp = total_fp = total_fn = total_tn = 0

        for pred_regions, gt_regions in zip(pred_region_list, gt_region_list):
            pred_has = region in set(pred_regions)
            gt_has = region in set(gt_regions)

            if pred_has and gt_has:
                total_tp += 1
            elif pred_has and not gt_has:
                total_fp += 1
            elif (not pred_has) and gt_has:
                total_fn += 1
            else:
                total_tn += 1

        p = total_tp / max(total_tp + total_fp, 1e-8)
        r = total_tp / max(total_tp + total_fn, 1e-8)
        f1 = 2 * p * r / max(p + r, 1e-8)
        acc = (total_tp + total_tn) / max(
            total_tp + total_fp + total_fn + total_tn, 1e-8
        )

        support = total_tp + total_fn
        pred_count = total_tp + total_fp

        out[region] = {
            "support": int(support),
            "pred_count": int(pred_count),
            "acc": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "tn": int(total_tn),
        }

        macro_ps.append(p)
        macro_rs.append(r)
        macro_f1s.append(f1)

    summary = {
        "per_region_label_macro_precision": float(np.mean(macro_ps)) if len(macro_ps) > 0 else 0.0,
        "per_region_label_macro_recall": float(np.mean(macro_rs)) if len(macro_rs) > 0 else 0.0,
        "per_region_label_macro_f1": float(np.mean(macro_f1s)) if len(macro_f1s) > 0 else 0.0,
        "per_region_label_num_regions": int(len(out)),
    }

    return summary, out


def compute_region_metrics(pred_region_list, gt_region_list):
    sample_ps, sample_rs, sample_f1s = [], [], []
    total_tp = total_fp = total_fn = 0

    for pred_regions, gt_regions in zip(pred_region_list, gt_region_list):
        pred = set(pred_regions)
        gt = set(gt_regions)

        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)

        p = tp / max(tp + fp, 1e-8)
        r = tp / max(tp + fn, 1e-8)
        f1 = 2 * p * r / max(p + r, 1e-8)

        sample_ps.append(p)
        sample_rs.append(r)
        sample_f1s.append(f1)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_p = total_tp / max(total_tp + total_fp, 1e-8)
    micro_r = total_tp / max(total_tp + total_fn, 1e-8)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)

    return {
        "region_sample_mean_precision": float(np.mean(sample_ps)),
        "region_sample_mean_recall": float(np.mean(sample_rs)),
        "region_sample_mean_f1": float(np.mean(sample_f1s)),
        "region_micro_precision": float(micro_p),
        "region_micro_recall": float(micro_r),
        "region_micro_f1": float(micro_f1),
        "region_tp": int(total_tp),
        "region_fp": int(total_fp),
        "region_fn": int(total_fn),
    }


def compute_geodesic_errors(pred_list, gt_list, geodesic_dist_npy):
    dist = np.load(geodesic_dist_npy, mmap_mode="r")

    if len(pred_list) == 0:
        raise RuntimeError("No SMPL-X samples for geodesic computation.")

    expected_n = len(pred_list[0])
    if dist.shape != (expected_n, expected_n):
        raise ValueError(
            f"geodesic_dist_npy shape mismatch: got {dist.shape}, "
            f"expected {(expected_n, expected_n)}."
        )

    fp_means = []
    fn_means = []

    for pred, gt in zip(pred_list, gt_list):
        pred_idx = np.where(pred > 0.5)[0]
        gt_idx = np.where(gt > 0.5)[0]

        fp_idx = np.setdiff1d(pred_idx, gt_idx, assume_unique=False)
        fn_idx = np.setdiff1d(gt_idx, pred_idx, assume_unique=False)

        if len(fp_idx) == 0:
            fp_means.append(0.0)
        elif len(gt_idx) == 0:
            fp_means.append(np.nan)
        else:
            d = dist[np.ix_(fp_idx, gt_idx)]
            d = np.asarray(d)
            d = np.where(np.isfinite(d), d, np.nan)
            nearest = np.nanmin(d, axis=1)
            fp_means.append(float(np.nanmean(nearest)))

        if len(fn_idx) == 0:
            fn_means.append(0.0)
        elif len(pred_idx) == 0:
            fn_means.append(np.nan)
        else:
            d = dist[np.ix_(fn_idx, pred_idx)]
            d = np.asarray(d)
            d = np.where(np.isfinite(d), d, np.nan)
            nearest = np.nanmin(d, axis=1)
            fn_means.append(float(np.nanmean(nearest)))

    return {
        "false_positive_geodesic": float(np.nanmean(fp_means)),
        "false_negative_geodesic": float(np.nanmean(fn_means)),
        "num_fp_geo_nan": int(np.isnan(fp_means).sum()),
        "num_fn_geo_nan": int(np.isnan(fn_means).sum()),
    }


def load_vertex_gt(path, gt_keys):
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    gt_vertices = {}
    for record in records:
        segment_id = str(record["segment_id"])
        for entry in record["smplx_vertices"]:
            for sentence_id, event in entry.items():
                start, end = event["frames"]
                for frame in range(start, end + 1):
                    key = (segment_id, str(sentence_id), frame)
                    if key in gt_vertices:
                        raise ValueError(f"Duplicate vertex GT: {key}")
                    gt_vertices[key] = event["vertices"]

    missing = gt_keys - gt_vertices.keys()
    extra = gt_vertices.keys() - gt_keys
    if missing or extra:
        raise ValueError(f"Spatial JSONL/vertex GT mismatch: {len(missing)} missing, {len(extra)} extra. "
                         f"Missing examples: {sorted(missing)[:5]}; extra examples: {sorted(extra)[:5]}")
    return gt_vertices


def make_vertex_mask(vertices, key):
    mask = np.zeros(NUM_VERTICES, dtype=np.float32)
    for vertex in vertices:
        if isinstance(vertex, bool) or not isinstance(vertex, int) or not 0 <= vertex < NUM_VERTICES:
            raise ValueError(f"Invalid SMPL-X vertex index {vertex!r}: {key}")
        mask[vertex] = 1.0
    return mask


def main():
    parser = argparse.ArgumentParser(description="HAPACT spatial localization evaluation")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--vertex", action="store_true")
    mode.add_argument("--region-label", action="store_true")
    parser.add_argument("--submission_path", default=str(DATA_DIR / "spatial" / "prediction.jsonl"))
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    gt_rows = load_jsonl(DATA_DIR / "spatial" / f"{args.split}.jsonl")
    if not gt_rows:
        raise ValueError(f"Empty spatial GT split: {args.split}")
    gt_keys = [sample_key(row) for row in gt_rows]
    if len(set(gt_keys)) != len(gt_keys):
        raise ValueError("Duplicate spatial GT keys in the benchmark JSONL")
    field = "pred_vertices" if args.vertex else "pred_regions"
    predictions = load_predictions(args.submission_path, set(gt_keys), field)
    regions = load_regions(DATA_DIR / "smplx_part_segmentation.json")

    if args.vertex:
        gt_vertex_map = load_vertex_gt(DATA_DIR / "opencv" / f"hapact_{args.split}_opencv.json", set(gt_keys))
        pred_list = [make_vertex_mask(predictions[key], key) for key in gt_keys]
        gt_list = [make_vertex_mask(gt_vertex_map[key], key) for key in gt_keys]
        metrics = compute_vertex_metrics(pred_list, gt_list)
        region_summary, region_detail = compute_body_region_vertex_metrics(pred_list, gt_list, regions, NUM_VERTICES)
        metrics.update(region_summary)
        geo = compute_geodesic_errors(pred_list, gt_list, DATA_DIR / "smplx_neutral_geodesic_dist.npy")
        metrics.update({
            "false_positive_geodesic_cm": geo["false_positive_geodesic"] * 100,
            "false_negative_geodesic_cm": geo["false_negative_geodesic"] * 100,
            "num_fp_geo_nan": geo["num_fp_geo_nan"],
            "num_fn_geo_nan": geo["num_fn_geo_nan"],
        })
        results = {"brief": {"num_matched": len(gt_keys), **metrics}, "per_body_region_metrics": region_detail}
        output_name = f"{args.split}_vertex_metrics.json"
    else:
        region_names = list(regions)
        valid_names = set(region_names)
        pred_list, gt_list = [], []
        for row, key in zip(gt_rows, gt_keys):
            pred, gt = predictions[key], row["gt_regions"]
            if any(not isinstance(r, str) or r not in valid_names for r in pred + gt):
                raise ValueError(f"Unknown region label: {key}")
            pred_list.append(pred)
            gt_list.append(gt)
        metrics = compute_region_metrics(pred_list, gt_list)
        region_summary, region_detail = compute_per_region_label_metrics(pred_list, gt_list, region_names)
        metrics.update(region_summary)
        results = {"brief": {"num_matched": len(gt_keys), **metrics}, "per_region_label_metrics": region_detail}
        output_name = f"{args.split}_region_label_metrics.json"

    save_path = Path(args.save_path) if args.save_path else Path(__file__).resolve().parent / "result" / "spatial" / output_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(json.dumps(results["brief"], indent=4, ensure_ascii=False))
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
