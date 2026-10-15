# Adapted from QV-M² (https://github.com/Zhuo-Cao/QV-M2).
# Copyright (c) 2024 Zhuo Cao. Licensed under the MIT License.

import numpy as np
from collections import OrderedDict, defaultdict
import json
import time
import copy
import multiprocessing as mp

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

from utils import compute_average_precision_detection, \
    compute_temporal_iou_batch_cross, compute_temporal_iou_batch_paired, load_jsonl, save_json, get_ap


def compute_average_precision_detection_wrapper(
        input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    qid, ground_truth, prediction = input_triple
    scores, far, md = compute_average_precision_detection(
        ground_truth, prediction, tiou_thresholds=tiou_thresholds)
    return qid, scores


def compute_ap_from_pr(tp, fp, npos):
    if npos <= 0:
        return 0.0

    tp = np.asarray(tp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    if len(tp) == 0:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / float(npos)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

    # VOC-style interpolated AP
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def window_to_frame_indices(window, fps=24.0):
    st, ed = float(window[0]), float(window[1])
    s_idx = int(round(st * fps))
    e_idx = int(round(ed * fps))
    if e_idx <= s_idx:
        e_idx = s_idx + 1
    return s_idx, e_idx


def boundary_distance_in_frames(pred_window, gt_window, fps=24.0):
    ps, pe = window_to_frame_indices(pred_window, fps=fps)
    gs, ge = window_to_frame_indices(gt_window, fps=fps)
    ds = abs(ps - gs)
    de = abs(pe - ge)
    return ds, de


def compute_average_precision_tolerance(ground_truth, prediction, deltas=np.arange(0, 11), fps=24.0):
    deltas = [int(d) for d in deltas]
    npos = len(ground_truth)

    if npos == 0:
        return np.zeros(len(deltas), dtype=float)

    # sort predictions by descending confidence
    prediction = sorted(prediction, key=lambda x: -float(x["score"]))

    ap_scores = []

    for delta in deltas:
        matched_gt = set()
        tp = []
        fp = []

        for pred in prediction:
            best_j = None
            best_key = None

            pred_window = [pred["t-start"], pred["t-end"]]

            for j, gt in enumerate(ground_truth):
                if j in matched_gt:
                    continue

                gt_window = [gt["t-start"], gt["t-end"]]
                ds, de = boundary_distance_in_frames(pred_window, gt_window, fps=fps)

                if ds <= delta and de <= delta:
                    key = (max(ds, de), ds + de, ds, de)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_j = j

            if best_j is not None:
                matched_gt.add(best_j)
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)

        ap_delta = compute_ap_from_pr(tp, fp, npos)
        ap_scores.append(ap_delta)

    return np.asarray(ap_scores, dtype=float)


def compute_average_precision_tolerance_wrapper(
        input_triple, deltas=np.arange(0, 11), fps=24.0):
    qid, ground_truth, prediction = input_triple
    scores = compute_average_precision_tolerance(
        ground_truth, prediction, deltas=deltas, fps=fps
    )
    return qid, scores


def compute_map_tolerance(
        submission,
        ground_truth,
        deltas=np.arange(0, 11),
        max_gt_windows=None,
        max_pred_windows=10,
        num_workers=8,
        chunksize=50,
        fps=24.0,
        return_per_query_ap=True):
    deltas = [int(d) for d in deltas]

    pred_qid2data = defaultdict(list)
    for d in submission:
        pred_windows = d["pred_relevant_windows"][:max_pred_windows] \
            if max_pred_windows is not None else d["pred_relevant_windows"]
        qid = d["qid"]
        for w in pred_windows:
            pred_qid2data[qid].append({
                "video-id": d["qid"],
                "t-start": float(w[0]),
                "t-end": float(w[1]),
                "score": float(w[2]),
            })

    gt_qid2data = defaultdict(list)
    for d in ground_truth:
        gt_windows = d["relevant_windows"][:max_gt_windows] \
            if max_gt_windows is not None else d["relevant_windows"]
        qid = d["qid"]
        for w in gt_windows:
            gt_qid2data[qid].append({
                "video-id": d["qid"],
                "t-start": float(w[0]),
                "t-end": float(w[1]),
            })

    qids = list(gt_qid2data.keys())

    from functools import partial
    compute_ap_from_triple = partial(
        compute_average_precision_tolerance_wrapper,
        deltas=deltas,
        fps=fps,
    )

    qid2ap_list = {}
    data_triples = [[qid, gt_qid2data[qid], pred_qid2data.get(qid, [])] for qid in qids]

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(compute_ap_from_triple, data_triples, chunksize=chunksize):
                qid2ap_list[qid] = scores
    else:
        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores

    if len(qid2ap_list) == 0:
        delta2map = {str(d): 0.0 for d in deltas}
        delta2map["average"] = 0.0
        out = {"mAP": delta2map}
        if return_per_query_ap:
            out["AP"] = {}
        return out

    ap_array = np.array([qid2ap_list[qid] for qid in qids], dtype=float)  
    map_per_delta = ap_array.mean(axis=0)

    delta2map = {str(d): float(f"{100.0 * map_per_delta[i]:.2f}") for i, d in enumerate(deltas)}
    delta2map["average"] = float(f"{100.0 * np.mean(map_per_delta):.2f}")

    out = {"mAP": delta2map}

    if return_per_query_ap:
        qid2delta2ap = {}
        for qid, scores in qid2ap_list.items():
            cur = {str(d): float(f"{100.0 * scores[i]:.2f}") for i, d in enumerate(deltas)}
            cur["average"] = float(f"{100.0 * np.mean(scores):.2f}")
            qid2delta2ap[qid] = cur
        out["AP"] = qid2delta2ap

    return out


def compute_mr_ap(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10),
                  max_gt_windows=None, max_pred_windows=10, num_workers=8, chunksize=50):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2data = defaultdict(list)
    for d in submission:
        pred_windows = d["pred_relevant_windows"][:max_pred_windows] \
            if max_pred_windows is not None else d["pred_relevant_windows"]
        qid = d["qid"]
        for w in pred_windows:
            pred_qid2data[qid].append({
                "video-id": d["qid"], 
                "t-start": w[0],
                "t-end": w[1],
                "score": w[2]
            })

    gt_qid2data = defaultdict(list)
    for d in ground_truth:
        gt_windows = d["relevant_windows"][:max_gt_windows] \
            if max_gt_windows is not None else d["relevant_windows"]
        qid = d["qid"]
        for w in gt_windows:
            gt_qid2data[qid].append({
                "video-id": d["qid"],
                "t-start": w[0],
                "t-end": w[1]
            })
    qid2ap_list = {}
    # start_time = time.time()
    data_triples = [[qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data]
    from functools import partial
    compute_ap_from_triple = partial(
        compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds)

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(compute_ap_from_triple, data_triples, chunksize=chunksize):
                qid2ap_list[qid] = scores
    else:
        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores

    # print(f"compute_average_precision_detection {time.time() - start_time:.2f} seconds.")
    ap_array = np.array(list(qid2ap_list.values()))  # (#queries, #thd)
    ap_thds = ap_array.mean(0)  # mAP at different IoU thresholds.
    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap["average"] = np.mean(ap_thds)
    # formatting
    iou_thd2ap = {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    return iou_thd2ap


def compute_mr_r1(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10)):
    """If a predicted segment has IoU >= iou_thd with one of the 1st GT segment, we define it positive"""
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2window = {
        d["qid"]: (d["pred_relevant_windows"][0][:2] if len(d["pred_relevant_windows"]) > 0 else None)
        for d in submission
    }
    # gt_qid2window = {d["qid"]: d["relevant_windows"][0] for d in ground_truth}
    gt_qid2window = {}
    pred_gt_iou = []
    for d in ground_truth:
        cur_gt_windows = d["relevant_windows"]
        cur_qid = d["qid"]
        if len(cur_gt_windows) == 0:
            continue
        if cur_qid not in pred_qid2window or pred_qid2window[cur_qid] is None:
            pred_gt_iou.append(0.0)
            continue
        if len(cur_gt_windows) > 0:  # select the GT window that has the highest IoU
            cur_ious = compute_temporal_iou_batch_cross(
                np.array([pred_qid2window[cur_qid]]), np.array(d["relevant_windows"])
            )[0]
            pred_gt_iou.append(float(np.max(cur_ious)))
    if len(pred_gt_iou) == 0:
        return {str(thd): 0.0 for thd in iou_thds}
    pred_gt_iou = np.array(pred_gt_iou, dtype=float)
    iou_thd2recall_at_one = {}
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(f"{np.mean(pred_gt_iou >= thd) * 100:.2f}")
    return iou_thd2recall_at_one


def compute_mr_miou(submission, ground_truth):
    pred_qid2window = {}
    for d in submission:
        qid = d["qid"]
        if len(d["pred_relevant_windows"]) == 0:
            continue
        pred_qid2window[qid] = d["pred_relevant_windows"][0][:2]

    gt_qid2window = {}
    for d in ground_truth:
        qid = d["qid"]
        if qid not in pred_qid2window:
            continue
        cur_gt_windows = d["relevant_windows"]
        if len(cur_gt_windows) == 0:
            continue
        cur_ious = compute_temporal_iou_batch_cross(
            np.array([pred_qid2window[qid]]), np.array(cur_gt_windows)
        )[0]
        gt_qid2window[qid] = cur_gt_windows[np.argmax(cur_ious)]

    qids = list(gt_qid2window.keys())
    if len(qids) == 0:
        return 0.0

    pred_windows = np.array([pred_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)

    return float(f"{np.mean(pred_gt_iou) * 100:.2f}")



def compute_mr_ap_by_gt(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10),
                        max_gt_windows=None, max_pred_windows=10, num_workers=8, chunksize=50):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2data = defaultdict(list)
    for d in submission:
        pred_windows = d["pred_relevant_windows"][:max_pred_windows] \
            if max_pred_windows is not None else d["pred_relevant_windows"]
        qid = d["qid"]
        for w in pred_windows:
            pred_qid2data[qid].append({
                "video-id": d["qid"], 
                "t-start": w[0],
                "t-end": w[1],
                "score": w[2]
            })

    gt_qid2data = defaultdict(list)
    for d in ground_truth:
        gt_windows = d["relevant_windows"][:max_gt_windows] \
            if max_gt_windows is not None else d["relevant_windows"]
        qid = d["qid"]
        for w in gt_windows:
            gt_qid2data[qid].append({
                "video-id": d["qid"],
                "t-start": w[0],
                "t-end": w[1]
            })

    # Calculate AP for each query
    qid2ap_list = {}
    data_triples = [[qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data]
    from functools import partial
    compute_ap_from_triple = partial(
        compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds)

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(compute_ap_from_triple, data_triples, chunksize=chunksize):
                qid2ap_list[qid] = scores
    else:
        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores

    # Group APs by GT count
    gt_count_to_aps = defaultdict(list)
    for qid, ap_scores in qid2ap_list.items():
        gt_count = len(gt_qid2data[qid])  # Number of GT windows for this query
        gt_count_to_aps[gt_count].append(ap_scores)

    total_queries = sum(len(aps) for aps in gt_count_to_aps.values())
    group_mean_ap_dict = {}
    group_mean_ap_dict["average"] = 0.0
    group_mean_ap_dict[">=3_tgt_mAP"] = 0.0
    group_mean_ap_dict["Generalized_mAP"] = 0.0
    count_3 = 0
    for gt_count, aps in gt_count_to_aps.items():
        aps = np.array(aps)  # (#queries@gt_count, #iou_thds)
        group_mean_ap = aps.mean(axis=0)
        weight = len(aps) / total_queries
        group_mean_ap_dict[f"{gt_count}_tgt_ap@iou"] = group_mean_ap
        group_mean_ap_dict[f"{gt_count}_tgt_mAP"] = group_mean_ap.mean()  # store the overall mean for each group
        group_mean_ap_dict["Generalized_mAP"] += group_mean_ap.mean() * weight
        if gt_count >= 3:
            count_3 += 1  # count queries whose GT size is >= 3
            group_mean_ap_dict[">=3_tgt_mAP"] += group_mean_ap.mean()
        
        group_mean_ap_dict[str(gt_count)] = group_mean_ap

    group_mean_ap_dict[">=3_tgt_mAP"] /= count_3 if count_3 > 0 else 1

    # Formatting
    group_mean_ap_dict = {
        k: (
            float(f"{100 * v:.2f}") if isinstance(v, (int, float)) else
            [float(f"{100 * elem:.2f}") for elem in v] if isinstance(v, np.ndarray) else v
        )
        for k, v in group_mean_ap_dict.items()
    }
    return group_mean_ap_dict


def match_topk_predictions_with_gts(preds, gts, iou_thd):
    """
    Match predicted windows with GT windows for the same query.
    Args:
      preds: list shaped [[start, end], ...], i.e., the top-k predictions.
      gts: list shaped [[start, end], ...], i.e., all GT windows.
      iou_thd: float, matching threshold.
    compute_temporal_iou_batch_cross is used to compute all IoUs at once.
    For each prediction we pick the unmatched GT window with the highest IoU,
    and if the IoU >= threshold we mark it matched and remove that GT.
    Returns the number of matches and the IoU list of all matches.
    """
    preds_arr = np.array(preds)  # shape: (k, 2)
    gts_arr = np.array(gts)      # shape: (m, 2)
    iou_matrix, _ = compute_temporal_iou_batch_cross(preds_arr, gts_arr)
    matched_gt = set()
    matched_ious = []
    
    for i in range(iou_matrix.shape[0]):
        best_iou = -1.0
        best_j = None
        for j in range(iou_matrix.shape[1]):
            if j in matched_gt:
                continue
            if iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j
        if best_iou >= iou_thd and best_j is not None:
            matched_gt.add(best_j)
            matched_ious.append(best_iou)
    return len(matched_gt), matched_ious

def compute_mr_all(submission, ground_truth, iou_thds=np.linspace(0.3, 0.95, 14)):
    """
    Compute Recall@1/2/3 and mIoU@1/2/3 based on top-k predictions.

    submission: list in the form
       {
         "qid": ...,
         "pred_relevant_windows": [[start, end, score], ...]  # sorted by confidence
       }
    ground_truth: list in the form
       {
         "qid": ...,
         "relevant_windows": [[gt_start, gt_end], ...]
       }
    iou_thds: IoU thresholds used for recall computation.

    Returns a tuple containing recall dictionaries at each k and mIoU@k.
    """
    # keep IoU thresholds to two decimals
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]

    # 1) collect top-3 prediction windows (only start/end) for each qid
    pred_qid2windows = {}
    for d in submission:
        qid = d["qid"]
        top3_preds = [item[:2] for item in d["pred_relevant_windows"][:3]]
        pred_qid2windows[qid] = top3_preds

    # 2) collect GT windows for each qid
    gt_qid2windows = {}
    for d in ground_truth:
        qid = d["qid"]
        gt_qid2windows[qid] = d["relevant_windows"]

    qids = list(pred_qid2windows.keys())

    # initialize recall@k dictionaries
    iou_thd2recall_at_1 = {}
    iou_thd2recall_at_2 = {}
    iou_thd2recall_at_3 = {}

    for thd in iou_thds:
        total_matched_1, total_gt_1 = 0, 0
        total_matched_2, total_gt_2 = 0, 0
        total_matched_3, total_gt_3 = 0, 0

        for qid in qids:
            gts = gt_qid2windows[qid]
            preds = pred_qid2windows[qid]  # up to 3 predictions

            # Recall@1: only evaluate queries with >=1 GT and >=1 prediction
            if len(gts) >= 1 and len(preds) >= 1:
                matched_1, _ = match_topk_predictions_with_gts(preds[:1], gts, thd)
                total_matched_1 += matched_1
                total_gt_1 += min(len(gts), 1)

            # Recall@2: only evaluate when both GTs and preds have at least 2 segments
            if len(gts) >= 2 and len(preds) >= 2:
                matched_2, _ = match_topk_predictions_with_gts(preds[:2], gts, thd)
                total_matched_2 += matched_2
                total_gt_2 += min(len(gts), 2)

            # Recall@3: only evaluate when both sides have at least 3 segments
            if len(gts) >= 3 and len(preds) >= 3:
                matched_3, _ = match_topk_predictions_with_gts(preds[:3], gts, thd)
                total_matched_3 += matched_3
                total_gt_3 += min(len(gts), 3)

        recall_at_1 = (total_matched_1 / total_gt_1 * 100) if total_gt_1 > 0 else 0.0
        recall_at_2 = (total_matched_2 / total_gt_2 * 100) if total_gt_2 > 0 else 0.0
        recall_at_3 = (total_matched_3 / total_gt_3 * 100) if total_gt_3 > 0 else 0.0

        iou_thd2recall_at_1[f"{thd:.2f}"] = float(f"{recall_at_1:.2f}")
        iou_thd2recall_at_2[f"{thd:.2f}"] = float(f"{recall_at_2:.2f}")
        iou_thd2recall_at_3[f"{thd:.2f}"] = float(f"{recall_at_3:.2f}")

    def compute_miou_at_k(k):
        ious_all = []
        for qid in qids:
            gts = gt_qid2windows[qid]
            preds = pred_qid2windows[qid][:k]
            if len(gts) >= k and len(preds) == k:
                _, matched_ious = match_topk_predictions_with_gts(preds, gts, iou_thd=-1.0)
                ious_all.extend(matched_ious)
        if len(ious_all) == 0:
            return 0.0
        return float(f"{np.mean(ious_all) * 100:.2f}")

    miou_at_1 = compute_miou_at_k(1)
    miou_at_2 = compute_miou_at_k(2)
    miou_at_3 = compute_miou_at_k(3)

    results = {
        "iou_thd2recall_at_1": iou_thd2recall_at_1,
        "iou_thd2recall_at_2": iou_thd2recall_at_2,
        "iou_thd2recall_at_3": iou_thd2recall_at_3,
        "mIoU@1": miou_at_1,
        "mIoU@2": miou_at_2,
        "mIoU@3": miou_at_3
    }
    return iou_thd2recall_at_1, iou_thd2recall_at_2, iou_thd2recall_at_3, miou_at_1, miou_at_2, miou_at_3

def get_window_len(window):
    return window[1] - window[0]


def get_data_by_range(submission, ground_truth, len_range):
    """ keep queries with ground truth window length in the specified length range.
    Args:
        submission:
        ground_truth:
        len_range: [min_l (int), max_l (int)]. the range is (min_l, max_l], i.e., min_l < l <= max_l
    """
    min_l, max_l = len_range
    if min_l == 0 and max_l == 150:  # min and max l in dataset
        return submission, ground_truth

    # only keep ground truth with windows in the specified length range
    # if multiple GT windows exists, we only keep the ones in the range
    ground_truth_in_range = []
    gt_qids_in_range = set()
    for d in ground_truth:
        rel_windows_in_range = [
            w for w in d["relevant_windows"] if min_l < get_window_len(w) <= max_l]
        if len(rel_windows_in_range) > 0:
            d = copy.deepcopy(d)
            d["relevant_windows"] = rel_windows_in_range
            ground_truth_in_range.append(d)
            gt_qids_in_range.add(d["qid"])

    # keep only submissions for ground_truth_in_range
    submission_in_range = []
    for d in submission:
        if d["qid"] in gt_qids_in_range:
            submission_in_range.append(copy.deepcopy(d))

    return submission_in_range, ground_truth_in_range


def eval_moment_retrieval(submission, ground_truth, verbose=True):
    length_ranges = [[0, 0.1], [0.1, 0.25], [0.25, 5], [0, 5]]
    range_names = ["short", "middle", "long", "full"]

    ret_metrics = {}
    for l_range, name in zip(length_ranges, range_names):
        if verbose:
            start_time = time.time()
        _submission, _ground_truth = get_data_by_range(submission, ground_truth, l_range)
        print(f"{name}: {l_range}, {len(_ground_truth)}/{len(ground_truth)}="
              f"{100*len(_ground_truth)/len(ground_truth):.2f} examples.")
        if len(_ground_truth) == 0:
            dummy_dict = {}
            for k in np.linspace(0.1, 0.5, 5):
                dummy_dict[k] = 0.
            dummy_dict['average'] = 0.
            ret_metrics[name] = {"MR-mAP": dummy_dict, "MR-R1": dummy_dict}
        else:
            map_iou_thds = np.linspace(0.1, 0.5, 5); mr_iou_thds = np.linspace(0.1, 0.5, 5)
            group_mean_ap_dict = compute_mr_ap_by_gt(_submission, _ground_truth, num_workers=8, chunksize=50, iou_thds=map_iou_thds)
            iou_thd2recall_at_1, iou_thd2recall_at_2, iou_thd2recall_at_3, mIoU_at_1, mIoU_at_2, mIoU_at_3 = compute_mr_all(_submission, _ground_truth, iou_thds=mr_iou_thds)
            map = compute_mr_ap(_submission, _ground_truth, iou_thds=np.array([0.1, 0.3, 0.5]))
            r1 = compute_mr_r1(_submission, _ground_truth, iou_thds=np.array([0.1, 0.3, 0.5]))
            miou = compute_mr_miou(_submission, _ground_truth)
            tolerance_scores = compute_map_tolerance(
                _submission,
                _ground_truth,
                deltas=np.arange(0, 6),
                max_pred_windows=10,
                num_workers=8,
                chunksize=50,
                fps=24.0,
                return_per_query_ap=True, 
            )
            ret_metrics[name] = {"MR-mIoU@1": mIoU_at_1,
                                 "MR-mIoU@2": mIoU_at_2,
                                 "MR-mIoU@3": mIoU_at_3,
                                 "MR-mAP": {
                                    "1_tgt_mAP": group_mean_ap_dict.get("1_tgt_mAP", 0.0),
                                    "2_tgt_mAP": group_mean_ap_dict.get("2_tgt_mAP", 0.0),
                                    ">=3_tgt_mAP": group_mean_ap_dict.get(">=3_tgt_mAP", 0.0),
                                    "Generalized_mAP": group_mean_ap_dict.get("Generalized_mAP", 0.0),
                                },
                                 "MR-R1": iou_thd2recall_at_1,
                                 "MR-R2": iou_thd2recall_at_2,
                                 "MR-R3": iou_thd2recall_at_3,
                                 "MR-mR1": round(np.mean(list(iou_thd2recall_at_1.values())), 2),
                                 "MR-mR2": round(np.mean(list(iou_thd2recall_at_2.values())), 2),
                                 "MR-mR3": round(np.mean(list(iou_thd2recall_at_3.values())), 2),
                                 "MR-mAP@0.1": map.get("0.1", 0.0),
                                 "MR-mAP@0.3": map.get("0.3", 0.0),
                                 "MR-mAP@0.5": map.get("0.5", 0.0),
                                 "MR-mAP-Avg": float(f"{np.mean([map.get('0.1', 0.0), map.get('0.3', 0.0), map.get('0.5', 0.0)]):.2f}"),
                                 "R1@0.1": r1.get("0.1", 0.0),
                                 "R1@0.3": r1.get("0.3", 0.0),
                                 "R1@0.5": r1.get("0.5", 0.0),
                                 "mIoU": miou,

                                 "Tolerance-mAP": tolerance_scores["mAP"],
                                 "Tolerance-mAP@0": tolerance_scores["mAP"].get("0", 0.0),
                                 "Tolerance-mAP@1": tolerance_scores["mAP"].get("1", 0.0),
                                 "Tolerance-mAP@2": tolerance_scores["mAP"].get("2", 0.0),
                                 "Tolerance-mAP@3": tolerance_scores["mAP"].get("3", 0.0),
                                 "Tolerance-mAP@4": tolerance_scores["mAP"].get("4", 0.0),
                                 "Tolerance-mAP@5": tolerance_scores["mAP"].get("5", 0.0),
                                 "Tolerance-mAP-Avg": tolerance_scores["mAP"].get("average", 0.0),
                                 }

    return ret_metrics


def compute_hl_hit1(qid2preds, qid2gt_scores_binary):
    qid2max_scored_clip_idx = {k: np.argmax(v["pred_saliency_scores"]) for k, v in qid2preds.items()}
    hit_scores = np.zeros((len(qid2preds), 3))
    qids = list(qid2preds.keys())
    for idx, qid in enumerate(qids):
        pred_clip_idx = qid2max_scored_clip_idx[qid]
        gt_scores_binary = qid2gt_scores_binary[qid]   # (#clips, 3)
        if pred_clip_idx < len(gt_scores_binary):
            hit_scores[idx] = gt_scores_binary[pred_clip_idx]
    # aggregate scores from 3 separate annotations (3 workers) by taking the max.
    # then average scores from all queries.
    hit_at_one = float(f"{100 * np.mean(np.max(hit_scores, 1)):.2f}")
    return hit_at_one


def compute_hl_ap(qid2preds, qid2gt_scores_binary, num_workers=8, chunksize=50):
    qid2pred_scores = {k: v["pred_saliency_scores"] for k, v in qid2preds.items()}
    ap_scores = np.zeros((len(qid2preds), 3))   # (#preds, 3)
    qids = list(qid2preds.keys())
    input_tuples = []
    for idx, qid in enumerate(qids):
        for w_idx in range(3):  # annotation score idx
            y_true = qid2gt_scores_binary[qid][:, w_idx]
            y_predict = np.array(qid2pred_scores[qid])
            input_tuples.append((idx, w_idx, y_true, y_predict))

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for idx, w_idx, score in pool.imap_unordered(
                    compute_ap_from_tuple, input_tuples, chunksize=chunksize):
                ap_scores[idx, w_idx] = score
    else:
        for input_tuple in input_tuples:
            idx, w_idx, score = compute_ap_from_tuple(input_tuple)
            ap_scores[idx, w_idx] = score

    # it's the same if we first average across different annotations, then average across queries
    # since all queries have the same #annotations.
    mean_ap = float(f"{100 * np.mean(ap_scores):.2f}")
    return mean_ap


def compute_ap_from_tuple(input_tuple):
    idx, w_idx, y_true, y_predict = input_tuple
    if len(y_true) < len(y_predict):
        y_predict = y_predict[:len(y_true)]
    elif len(y_true) > len(y_predict):
        _y_predict = np.zeros(len(y_true))
        _y_predict[:len(y_predict)] = y_predict
        y_predict = _y_predict

    score = get_ap(y_true, y_predict)
    return idx, w_idx, score


def mk_gt_scores(gt_data, clip_length=2):
    """gt_data, dict, """
    num_clips = int(gt_data["duration"] / clip_length)
    saliency_scores_full_video = np.zeros((num_clips, 3))
    relevant_clip_ids = np.array(gt_data["relevant_clip_ids"])  # (#relevant_clip_ids, )
    saliency_scores_relevant_clips = np.array(gt_data["saliency_scores"])  # (#relevant_clip_ids, 3)
    saliency_scores_full_video[relevant_clip_ids] = saliency_scores_relevant_clips
    return saliency_scores_full_video  # (#clips_in_video, 3)  the scores are in range [0, 4]


def eval_highlight(submission, ground_truth, verbose=True):
    """
    Args:
        submission:
        ground_truth:
        verbose:
    """
    qid2preds = {d["qid"]: d for d in submission}
    qid2gt_scores_full_range = {d["qid"]: mk_gt_scores(d) for d in ground_truth}  # scores in range [0, 4]
    # gt_saliency_score_min: int, in [0, 1, 2, 3, 4]. The minimum score for a positive clip.
    gt_saliency_score_min_list = [2, 3, 4]
    saliency_score_names = ["Fair", "Good", "VeryGood"]
    highlight_det_metrics = {}
    for gt_saliency_score_min, score_name in zip(gt_saliency_score_min_list, saliency_score_names):
        start_time = time.time()
        qid2gt_scores_binary = {
            k: (v >= gt_saliency_score_min).astype(float)
            for k, v in qid2gt_scores_full_range.items()}  # scores in [0, 1]
        hit_at_one = compute_hl_hit1(qid2preds, qid2gt_scores_binary)
        mean_ap = compute_hl_ap(qid2preds, qid2gt_scores_binary)
        highlight_det_metrics[f"HL-min-{score_name}"] = {"HL-mAP": mean_ap, "HL-Hit1": hit_at_one}
        if verbose:
            print(f"Calculating highlight scores with min score {gt_saliency_score_min} ({score_name})")
            print(f"Time cost {time.time() - start_time:.2f} seconds")
    return highlight_det_metrics


def eval_submission(submission, ground_truth, verbose=True, match_number=True):
    """
    Args:
        submission: list(dict), each dict is {
            qid: str,
            query: str,
            vid: str,
            pred_relevant_windows: list([st, ed]),
            pred_saliency_scores: list(float), len == #clips in video.
                i.e., each clip in the video will have a saliency score.
        }
        ground_truth: list(dict), each dict is     {
          "qid": 7803,
          "query": "Man in gray top walks from outside to inside.",
          "duration": 150,
          "vid": "RoripwjYFp8_360.0_510.0",
          "relevant_clip_ids": [13, 14, 15, 16, 17]
          "saliency_scores": [[4, 4, 2], [3, 4, 2], [2, 2, 3], [2, 2, 2], [0, 1, 3]]
               each sublist corresponds to one clip in relevant_clip_ids.
               The 3 elements in the sublist are scores from 3 different workers. The
               scores are in [0, 1, 2, 3, 4], meaning [Very Bad, ..., Good, Very Good]
        }
        verbose:
        match_number:

    Returns:

    """
    pred_qids = set([e["qid"] for e in submission])
    gt_qids = set([e["qid"] for e in ground_truth])
    if match_number:
        assert pred_qids == gt_qids, \
            f"qids in ground_truth and submission must match. " \
            f"use `match_number=False` if you wish to disable this check"
    else:  # only leave the items that exists in both submission and ground_truth
        shared_qids = pred_qids.intersection(gt_qids)
        submission = [e for e in submission if e["qid"] in shared_qids]
        ground_truth = [e for e in ground_truth if e["qid"] in shared_qids]

    eval_metrics = {}
    eval_metrics_brief = OrderedDict()
    if "pred_relevant_windows" in submission[0]:
        moment_ret_scores = eval_moment_retrieval(
            submission, ground_truth, verbose=verbose)
        eval_metrics.update(moment_ret_scores)
        moment_ret_scores_brief = {
            "MR-full-mAP": moment_ret_scores["full"]["MR-mAP"]["Generalized_mAP"],
            "MR-full-mAP@1_tgt": moment_ret_scores["full"]["MR-mAP"]["1_tgt_mAP"],
            "MR-full-mAP@2_tgt": moment_ret_scores["full"]["MR-mAP"]["2_tgt_mAP"],
            "MR-full-mAP@3+tgt": moment_ret_scores["full"]["MR-mAP"][">=3_tgt_mAP"],
            "MR-full-mIoU@1": moment_ret_scores["full"]["MR-mIoU@1"],
            "MR-full-mIoU@2": moment_ret_scores["full"]["MR-mIoU@2"],
            "MR-full-mIoU@3": moment_ret_scores["full"]["MR-mIoU@3"],
            "MR-full-mR1": moment_ret_scores["full"]["MR-mR1"],
            "MR-full-mR2": moment_ret_scores["full"]["MR-mR2"],
            "MR-full-mR3": moment_ret_scores["full"]["MR-mR3"],
            }
        eval_metrics_brief.update(
            sorted([(k, v) for k, v in moment_ret_scores_brief.items()], key=lambda x: x[0]))

    # sort by keys
    final_eval_metrics = OrderedDict()
    final_eval_metrics["brief"] = eval_metrics_brief
    final_eval_metrics.update(sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0]))
    return final_eval_metrics


def snap_timestamp_to_grid(ts, fps=24.0):
    return round(float(ts) * fps) / fps


def snap_window_to_grid(st, ed, duration=None, fps=24.0):
    st = float(st)
    ed = float(ed)

    s_idx = int(round(st * fps))
    e_idx = int(round(ed * fps))

    if duration is not None:
        max_idx = int(round(float(duration) * fps))
        if max_idx <= 0:
            return 0.0, 1.0 / fps

        s_idx = min(max(s_idx, 0), max(0, max_idx - 1))
        e_idx = min(max(e_idx, 1), max_idx)

    else:
        s_idx = max(s_idx, 0)
        e_idx = max(e_idx, 0)

    if e_idx <= s_idx:
        if duration is not None:
            if s_idx + 1 <= max_idx:
                e_idx = s_idx + 1
            else:
                s_idx = max(0, max_idx - 1)
                e_idx = max_idx
        else:
            e_idx = s_idx + 1

    st_new = s_idx / fps
    ed_new = e_idx / fps
    return st_new, ed_new


def clean_submission_item(item, duration=None, fps=24.0):
    out = dict(item)
    out.setdefault("pred_relevant_windows", [])
    cleaned = []
    for w in out["pred_relevant_windows"]:
        if len(w) == 2:
            st, ed, score = float(w[0]), float(w[1]), 1.0
        else:
            st, ed, score = float(w[0]), float(w[1]), float(w[2])
        st, ed = snap_window_to_grid(st, ed, duration=duration, fps=fps)
        cleaned.append([st, ed, score])
    cleaned.sort(key=lambda x: -x[2])
    out["pred_relevant_windows"] = cleaned
    return out


def clean_gt_item(item, fps=24.0):
    duration = float(item.get("duration", 0.0))
    relevant_windows = []
    for w in item.get("relevant_windows", []):
        st, ed = snap_window_to_grid(float(w[0]), float(w[1]),
                                     duration=duration if duration > 0 else None,
                                     fps=fps
        )
        relevant_windows.append([st, ed])

    out = {
        "qid": item["qid"],
        "query": item.get("query", ""),
        "vid": item.get("vid", ""),
        "duration": duration,
        "relevant_windows": relevant_windows,
    }
    return out


def load_gt_by_split(data_root, split):
    if split == "total":
        gt = []
        for s in ["train", "val", "test"]:
            gt.extend(load_jsonl(os.path.join(data_root, f"{s}.jsonl")))
        return gt
    return load_jsonl(os.path.join(data_root, f"{split}.jsonl"))


def eval_main():
    import argparse
    parser = argparse.ArgumentParser(description="HAPACT temporal localization evaluation")
    parser.add_argument("--submission_path", type=str, default=None, help="path to generated prediction file")
    parser.add_argument("--save_path", type=str, default=None, help="path to save the results")
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=str(Path(__file__).resolve().parent / "data" / "temporal"))
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--not_verbose", action="store_true")
    args = parser.parse_args()

    verbose = not args.not_verbose
    submission_path = os.path.join(args.data_root, "prediction.jsonl") if args.submission_path is None else args.submission_path
    save_path = str(Path(__file__).resolve().parent / "result" / "temporal" / f"{args.split}_metrics.json") if args.save_path is None else args.save_path

    ground_truth = [clean_gt_item(x, fps=args.fps) for x in load_gt_by_split(args.data_root, args.split)]
    qid2duration = {x["qid"]: x["duration"] for x in ground_truth}

    submission_raw = load_jsonl(submission_path)
    
    pred_qids = [item["qid"] for item in submission_raw]
    gt_qids = {item["qid"] for item in ground_truth}

    if len(pred_qids) != len(set(pred_qids)):
        raise ValueError("Duplicate prediction qids found.")

    missing = gt_qids - set(pred_qids)
    extra = set(pred_qids) - gt_qids
    if missing or extra:
        raise ValueError(
            f"Prediction/GT qid mismatch: {len(missing)} missing, {len(extra)} extra. "
            f"Missing examples: {sorted(missing)[:5]}; extra examples: {sorted(extra)[:5]}"
        )
    
    submission = [
        clean_submission_item(x, duration=qid2duration.get(x["qid"], None), fps=args.fps)
        for x in submission_raw
    ]
    
    results = eval_moment_retrieval(
        submission=submission,
        ground_truth=ground_truth
    )

    if verbose:
        print(json.dumps(results, indent=4))
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    with open(save_path, "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == '__main__':
    eval_main()
