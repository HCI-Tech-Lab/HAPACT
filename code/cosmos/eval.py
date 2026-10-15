import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


CODE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CODE_DIR / "data"
OUTPUT_DIR = CODE_DIR / "result" / "cosmos"


def save_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")



def resolve_image_path(imgname, image_root, strip_image_prefix=None):
    imgname = str(imgname)

    if strip_image_prefix is not None:
        prefix = strip_image_prefix.rstrip("/")
        if imgname.startswith(prefix + "/"):
            imgname = imgname[len(prefix) + 1:]
        elif imgname.startswith("./" + prefix + "/"):
            imgname = imgname[len("./" + prefix + "/"):]

    img_path = Path(imgname)
    if not img_path.is_absolute():
        img_path = Path(image_root) / img_path

    return img_path.resolve()



def extract_region_names_from_prompt(prompt):
    marker = "The possible body regions are:"
    if marker not in prompt:
        return []

    text = prompt.split(marker, 1)[1].split(".", 1)[0].strip()
    return [x.strip() for x in text.split(",") if x.strip()]



def parse_regions_from_output(text, region_names):
    """
    Output example:
      The affected body regions are right arm, hips, and right shoulder.
    """
    lower = text.lower().strip()

    # remove common generation noise
    lower = lower.replace("\n", " ")
    lower = re.sub(r"\s+", " ", lower)

    # If model gives extra explanation, focus after target phrase when possible.
    idx = lower.find("the affected body region")
    if idx >= 0:
        lower = lower[idx:]

    pred = []

    # Prevents "right arm" matching before "right upper arm" if both exist.
    for region in sorted(region_names, key=len, reverse=True):
        r = region.lower()
        # region names include spaces, slash-free; exact substring is okay here.
        if re.search(rf"(?<![a-zA-Z]){re.escape(r)}(?![a-zA-Z])", lower):
            pred.append(region)

    pred = sorted(set(pred), key=lambda x: region_names.index(x) if x in region_names else 999)

    return pred



def make_image_only_prompt(original_prompt):
    marker = "The possible body regions are:"
    if marker not in original_prompt:
        raise ValueError(
            "Cannot build image_only prompt because candidate region marker was not found."
        )

    region_text = original_prompt.split(marker, 1)[1].split(".", 1)[0].strip()

    return (
        "Identify the body regions physically affected by the impact in the image. "
        f"{marker} {region_text}. "
        "Answer with only the affected body region names."
    )



def align_special_tokens(model, processor):
    model.config.tie_word_embeddings = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.tie_word_embeddings = False
    if hasattr(model.config, "language_config"):
        model.config.language_config.tie_word_embeddings = False
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        model.language_model.config.tie_word_embeddings = False

    tok = processor.tokenizer
    model.config.eos_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.bos_token_id = tok.bos_token_id

    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.bos_token_id = tok.bos_token_id



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="nvidia/Cosmos-Reason2-8B")
    parser.add_argument("--resume", default=str(OUTPUT_DIR / "best_region_checkpoint"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--image_root", default=str(DATA_DIR / "images"))
    parser.add_argument("--save_prediction_path", default=str(OUTPUT_DIR / "prediction.jsonl"))
    parser.add_argument("--save_metrics_path", default=None)
    parser.add_argument("--input_modality", choices=["full", "text_only", "image_only"], default="full")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--strip_image_prefix", default=None)
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=str(DATA_DIR / "spatial" / f"{args.split}.jsonl"), split="train")
    if len(dataset) == 0:
        raise ValueError(f"Empty spatial split: {args.split}")
    region_names = extract_region_names_from_prompt(dataset[0]["prompt"])
    if not region_names:
        raise ValueError("No region choices found in the benchmark prompt.")

    print("num eval samples:", len(dataset))
    print("regions:", region_names)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    print("Loading base model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    align_special_tokens(model, processor)

    print("Loading LoRA adapter:", args.resume)
    model = PeftModel.from_pretrained(model, args.resume)
    model.eval()

    pred_records = []
    for ex in dataset:
        content = []
        if args.input_modality != "text_only":
            image_path = resolve_image_path(
                ex["imgname"], image_root=args.image_root,
                strip_image_prefix=args.strip_image_prefix,
            )
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            content.append({"type": "image", "image": str(image_path)})

        if args.input_modality == "image_only":
            prompt_text = make_image_only_prompt(ex["prompt"])
        else:
            prompt_text = ex["prompt"]
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        pred_regions = parse_regions_from_output(raw_output, region_names)

        pred_records.append({
            "segment_id": ex["segment_id"],
            "sentence_id": ex["sentence_id"],
            "target_frame": ex["target_frame"],
            "pred_regions": pred_regions,
        })

    prediction_path = Path(args.save_prediction_path)
    metrics_path = Path(args.save_metrics_path) if args.save_metrics_path else OUTPUT_DIR / f"{args.split}_metrics.json"
    save_jsonl(prediction_path, pred_records)

    cmd = [
        sys.executable,
        str(CODE_DIR / "eval_spatial.py"),
        "--split", args.split,
        "--region-label",
        "--submission_path", str(prediction_path),
        "--save_path", str(metrics_path),
    ]
    subprocess.run(cmd, check=True)
    print("saved predictions:", prediction_path)
    print("saved metrics:", metrics_path)


if __name__ == "__main__":
    main()
