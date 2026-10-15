import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"
SPLITS = ("train", "val", "test")


def checksum(data):
    packed = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def format_json(value, indent=0):
    space = " " * indent
    inside = " " * (indent + 2)
    if isinstance(value, dict):
        if not value:
            return "{}"
        entries = []
        for key, item in value.items():
            entries.append(f"{inside}{json.dumps(key, ensure_ascii=False)}: {format_json(item, indent + 2)}")
        return "{\n" + ",\n".join(entries) + f"\n{space}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(item is None or isinstance(item, (str, int, float, bool)) for item in value):
            return json.dumps(value, ensure_ascii=False)
        entries = [f"{inside}{format_json(item, indent + 2)}" for item in value]
        return "[\n" + ",\n".join(entries) + f"\n{space}]"
    return json.dumps(value, ensure_ascii=False)


def apply_edits(records, edits):
    for record in records:
        patch = edits.get(record["segment_id"])
        if not patch:
            continue
        if "t" in patch:
            record["relevant_frames"] = patch["t"]

        events = record["smplx_vertices"]
        for index, new_key in patch.get("k", []):
            events[index] = {new_key: next(iter(events[index].values()))}
        for index, start, end in patch.get("f", []):
            next(iter(events[index].values()))["frames"] = [start, end]
        for index, vertices in patch.get("v", []):
            next(iter(events[index].values()))["vertices"] = vertices
        for index in sorted(patch.get("d", []), reverse=True):
            del events[index]


def to_zero_based(records):
    for record in records:
        for intervals in record["relevant_frames"].values():
            for interval in intervals:
                interval[0] -= 1
                interval[1] -= 1
        for entry in record["smplx_vertices"]:
            for event in entry.values():
                event["frames"] = [frame - 1 for frame in event["frames"]]


def temporal_rows(records):
    for record in records:
        windows = [[start / 24, (end + 1) / 24]
                   for intervals in record["relevant_frames"].values()
                   for start, end in intervals]
        yield {
            "qid": record["segment_id"],
            "query": record["description"],
            "duration": record["total_frames"] / 24,
            "relevant_windows": windows,
        }


def spatial_rows(records, segmentation):
    parts = [(re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower().replace("fore arm", "forearm"), set(vertices))
             for name, vertices in segmentation.items()]
    region_names = ", ".join(name for name, _ in parts)

    for record in records:
        segment_id = record["segment_id"]
        frame_rows = []
        for entry in record["smplx_vertices"]:
            for sentence_id, event in entry.items():
                affected = set(event["vertices"])
                scores = [(name, len(affected & vertices) / len(vertices)) for name, vertices in parts]
                regions = [name for name, score in scores if score >= 0.3]
                if not regions:
                    regions = [max(scores, key=lambda item: item[1])[0]]

                if len(regions) == 1:
                    answer = f"The affected body region is {regions[0]}."
                elif len(regions) == 2:
                    answer = f"The affected body regions are {regions[0]} and {regions[1]}."
                else:
                    answer = f"The affected body regions are {', '.join(regions[:-1])}, and {regions[-1]}."

                query = record["sub_sentences"][sentence_id]
                prompt = (
                    "You are observing a movie frame. "
                    f'The impact query is "{query}". '
                    "Select all body regions physically affected by the impact in this frame. "
                    f"The possible body regions are: {region_names}.\n"
                    "Answer with only the selected body-region labels, separated by commas."
                )
                
                for frame in range(event["frames"][0], event["frames"][1] + 1):
                    frame_rows.append({
                        "segment_id": segment_id,
                        "sentence_id": sentence_id,
                        "query": query,
                        "target_frame": frame,
                        "imgname": f"./{segment_id}_{frame:06d}.png",
                        "prompt": prompt,
                        "answer": answer,
                        "gt_regions": regions,
                        "gt_vertices": event["vertices"],
                    })
        yield from sorted(frame_rows, key=lambda row: row["target_frame"])


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--spatial", action="store_true")
    args = parser.parse_args()

    with np.load(DATA_DIR / "opencv_mapping.npz", allow_pickle=False) as archive:
        mapping = json.loads(archive["patch"].tobytes().decode("utf-8"))
    if mapping["version"] != 1:
        raise ValueError("Unsupported mapping version")

    if args.spatial:
        with (DATA_DIR / "smplx_part_segmentation.json").open("r", encoding="utf-8") as f:
            segmentation = json.load(f)

    for split in SPLITS:
        input_path = DATA_DIR / f"hapact_{split}.json"
        output_path = DATA_DIR / "opencv" / f"hapact_{split}_opencv.json"
        with input_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        config = mapping["splits"][split]
        if checksum(records) != config["sha256"]:
            raise ValueError(f"Source annotations differ from the expected {split} split: {input_path}")
        apply_edits(records, config["edits"])
        to_zero_based(records)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(format_json(records))
            f.write("\n")
        print(f"Saved {output_path}")

        if args.temporal:
            save_jsonl(DATA_DIR / "temporal" / f"{split}.jsonl", temporal_rows(records))
        if args.spatial:
            save_jsonl(DATA_DIR / "spatial" / f"{split}.jsonl", spatial_rows(records, segmentation))


if __name__ == "__main__":
    main()
