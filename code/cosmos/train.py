import argparse
from pathlib import Path
import gc

import sys
import torch
import json
import subprocess
import shutil
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
from trl import SFTConfig, SFTTrainer


CODE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CODE_DIR / "data"
OUTPUT_DIR = CODE_DIR / "result" / "cosmos"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="nvidia/Cosmos-Reason2-8B")
    parser.add_argument("--train_jsonl", nargs="+", default=[str(DATA_DIR / "spatial" / "train.jsonl")])
    parser.add_argument("--eval_jsonl", default=str(DATA_DIR / "spatial" / "val.jsonl"))
    parser.add_argument("--image_root", default=str(DATA_DIR / "images"))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))

    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--input_modality", choices=["full", "text_only", "image_only"], default="full")

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=400)
    parser.add_argument("--save_steps", type=int, default=400)
    parser.add_argument("--save_total_limit", type=int, default=50)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--strip_image_prefix", default=None)

    # region-F1 checkpoint sweep
    parser.add_argument("--region_eval_only", action="store_true")
    parser.add_argument("--region_eval_script", default="eval.py")
    parser.add_argument("--region_metric_name", default="region_sample_mean_f1")
    parser.add_argument("--region_eval_limit", type=int, default=-1)
    parser.add_argument("--region_max_new_tokens", type=int, default=64)
    parser.add_argument("--force_region_eval", action="store_true")
    parser.add_argument("--best_region_dir_name", default="best_region_checkpoint")

    parser.add_argument("--no_train_eval", action="store_true")
    parser.add_argument("--run_region_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    return parser.parse_args()



def normalize_jsonl_paths(jsonl_paths):
    if isinstance(jsonl_paths, (str, Path)):
        jsonl_paths = [jsonl_paths]
    paths = [Path(path).expanduser() for path in jsonl_paths]
    if not paths:
        raise ValueError("At least one JSONL path is required.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following JSONL files do not exist:\n"
            + "\n".join(f"  {path}" for path in missing)
        )
    return paths


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


def build_messages(example, image_root, strip_image_prefix=None, input_modality="full"):
    if input_modality not in ["full", "text_only", "image_only"]:
        raise ValueError(f"Unknown input_modality: {input_modality}")

    images = []
    content = []

    if input_modality != "text_only":
        image_path = resolve_image_path(
            example["imgname"],
            image_root=image_root,
            strip_image_prefix=strip_image_prefix,
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}\n"
                f"imgname={example['imgname']}\n"
                f"image_root={image_root}"
            )

        images = [str(image_path)]
        content.append({"type": "image"})

    if input_modality == "image_only":
        prompt_text = make_image_only_prompt(example["prompt"])
    else:
        prompt_text = example["prompt"]

    content.append({"type": "text", "text": prompt_text})

    return {
        "images": images,
        "prompt": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": example["answer"]},
                ],
            }
        ],
    }


def load_jsonl_datasets(jsonl_paths, split_name):
    paths = normalize_jsonl_paths(jsonl_paths)
    datasets = []
    for index, path in enumerate(paths, 1):
        dataset = load_dataset(
            "json",
            data_files={"train": str(path)},
            split="train",
        )
        print(
            f"[{split_name}] file {index}/{len(paths)}: "
            f"{path.resolve()} | samples={len(dataset)}"
        )
        datasets.append(dataset)
    if len(datasets) == 1:
        merged = datasets[0]
    else:
        merged = concatenate_datasets(datasets)
    print(f"[{split_name}] merged files:", len(paths))
    print(f"[{split_name}] merged raw size:", len(merged))
    return merged


def report_duplicate_imgnames(dataset, split_name):
    if "imgname" not in dataset.column_names:
        return
    seen = set()
    duplicates = []
    for imgname in dataset["imgname"]:
        value = str(imgname).replace("\\", "/")
        if value in seen:
            duplicates.append(value)
        else:
            seen.add(value)
    if duplicates:
        print(
            f"[{split_name}] WARNING: "
            f"{len(duplicates)} duplicate imgname rows were found after merging."
        )
        for value in duplicates[:20]:
            print(f"[{split_name}] duplicate:", value)
    else:
        print(f"[{split_name}] duplicate imgname rows: 0")



def prepare_dataset(jsonl_paths, image_root, strip_image_prefix=None, split_name="train", input_modality="full"):
    dataset = load_jsonl_datasets(
        jsonl_paths=jsonl_paths,
        split_name=split_name,
    )
    print(f"[{split_name}] columns:", dataset.column_names)
    report_duplicate_imgnames(dataset, split_name)
    dataset = dataset.map(
        lambda ex: build_messages(
            ex,
            image_root=image_root,
            strip_image_prefix=strip_image_prefix,
            input_modality=input_modality,
        ),
        num_proc=1,
        desc=f"Building {split_name} messages",
    )
    print(f"[{split_name}] processed columns:", dataset.column_names)
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n[{split_name}] sample prompt:", sample["prompt"])
        print(f"[{split_name}] sample completion:", sample["completion"])
        print(f"[{split_name}] sample images:", sample["images"])
    cols_to_keep = {"images", "prompt", "completion"}
    cols_to_remove = [
        column
        for column in dataset.column_names
        if column not in cols_to_keep
    ]
    dataset = dataset.remove_columns(cols_to_remove)
    print(
        f"[{split_name}] processed columns after cleanup:",
        dataset.column_names,
    )
    return dataset


def checkpoint_step(path: Path):
    try:
        return int(path.name.split("-")[-1])
    except Exception:
        return -1


def find_checkpoints(output_dir):
    output_dir = Path(output_dir)
    ckpts = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    ckpts = sorted(ckpts, key=checkpoint_step)
    return ckpts


def run_region_eval_sweep(args):
    output_dir = Path(args.output_dir)
    ckpts = find_checkpoints(output_dir)

    if len(ckpts) == 0:
        print("[REGION EVAL] No checkpoints found.", flush=True)
        return None

    if args.region_eval_limit > 0:
        raise ValueError("The spatial benchmark requires predictions for the full validation split.")
    sweep_dir = output_dir / "region_eval_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"[REGION EVAL] Found {len(ckpts)} checkpoints.", flush=True)
    print(f"[REGION EVAL] split: val", flush=True)
    print(f"[REGION EVAL] metric: {args.region_metric_name}", flush=True)

    rows = []
    best = None

    for ckpt in ckpts:
        step = checkpoint_step(ckpt)
        ckpt_eval_dir = sweep_dir / ckpt.name
        ckpt_eval_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = ckpt_eval_dir / "metrics.json"
        pred_path = ckpt_eval_dir / "predictions.jsonl"

        if metrics_path.exists() and pred_path.exists() and not args.force_region_eval:
            print(f"[REGION EVAL] Reusing existing metrics: {metrics_path}", flush=True)
        else:
            eval_script = Path(args.region_eval_script)
            if not eval_script.is_absolute():
                eval_script = Path(__file__).resolve().parent / eval_script

            cmd = [
                sys.executable,
                str(eval_script),
                "--model_name", args.model_name,
                "--resume", str(ckpt),
                "--split", "val",
                "--image_root", str(args.image_root),
                "--save_prediction_path", str(pred_path),
                "--save_metrics_path", str(metrics_path),
                "--max_new_tokens", str(args.region_max_new_tokens),
                "--input_modality", str(args.input_modality),
            ]

            if args.strip_image_prefix is not None:
                cmd += ["--strip_image_prefix", str(args.strip_image_prefix)]

            print("[REGION EVAL] Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        brief = metrics["brief"]
        if args.region_metric_name not in brief:
            raise KeyError(
                f"{args.region_metric_name} not found in {metrics_path}. "
                f"Available keys: {list(brief.keys())}"
            )

        score = float(brief[args.region_metric_name])
        row = {
            "checkpoint": str(ckpt),
            "step": step,
            "metric_name": args.region_metric_name,
            "score": score,
            "metrics_path": str(metrics_path),
            "prediction_path": str(pred_path),
            "brief": brief,
        }
        rows.append(row)

        print(
            f"[REGION EVAL] {ckpt.name}: "
            f"{args.region_metric_name}={score:.6f}, "
            f"region_micro_f1={brief.get('region_micro_f1')}, "
            f"region_sample_mean_f1={brief.get('region_sample_mean_f1')}",
            flush=True,
        )

        if best is None or score > best["score"]:
            best = row

    summary_path = sweep_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metric_name": args.region_metric_name,
                "best": best,
                "all": rows,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    best_dst = output_dir / args.best_region_dir_name
    if best_dst.exists():
        shutil.rmtree(best_dst)

    best_src = Path(best["checkpoint"])
    print(f"[REGION EVAL] Best checkpoint: {best_src}", flush=True)
    print(f"[REGION EVAL] Best {args.region_metric_name}: {best['score']:.6f}", flush=True)
    print(f"[REGION EVAL] Copying to: {best_dst}", flush=True)

    shutil.copytree(best_src, best_dst)
    shutil.copy2(best["metrics_path"], output_dir / "best_region_metrics.json")
    shutil.copy2(best["prediction_path"], output_dir / "prediction.jsonl")

    print(f"[REGION EVAL] Saved summary: {summary_path}", flush=True)
    print(f"[REGION EVAL] Saved best metrics: {output_dir / 'best_region_metrics.json'}", flush=True)

    return best


def main():
    args = parse_args()

    if args.region_eval_only:
        best = run_region_eval_sweep(args)
        if best is not None:
            print("[DONE] Best region checkpoint:", best["checkpoint"])
            print("[DONE] Best score:", best["score"])
        else:
            print("[DONE] Region eval skipped: no checkpoint found.")
        return

    train_dataset = prepare_dataset(
        args.train_jsonl,
        image_root=args.image_root,
        strip_image_prefix=args.strip_image_prefix,
        split_name="train",
        input_modality=args.input_modality,
    )

    eval_dataset = None
    if (args.eval_jsonl is not None) and (not args.no_train_eval):
        eval_dataset = prepare_dataset(
            args.eval_jsonl,
            image_root=args.image_root,
            strip_image_prefix=args.strip_image_prefix,
            split_name="eval",
            input_modality=args.input_modality,
        )

    if args.dry_run:
        print("Dry run finished.")
        return

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,

        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        optim="adamw_8bit" if args.load_in_4bit else "adamw_torch",

        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        eval_strategy="no" if args.no_train_eval or eval_dataset is None else "steps",
        eval_steps=args.eval_steps if (eval_dataset is not None and not args.no_train_eval) else None,

        load_best_model_at_end=False,
        report_to="tensorboard",
        max_length=None,
        completion_only_loss=True,

        bf16=args.bf16,
        fp16=args.fp16,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processor,
    )

    base_collator = trainer.data_collator

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    print("image_token_id:", image_token_id, flush=True)
    print("video_token_id:", video_token_id, flush=True)

    def fixed_data_collator(features):
        batch = base_collator(features)

        input_ids = batch["input_ids"]

        mm = torch.zeros(
            input_ids.shape,
            dtype=torch.int32,
            device=input_ids.device,
        )

        if image_token_id is not None and image_token_id >= 0:
            mm[input_ids == image_token_id] = 1

        if video_token_id is not None and video_token_id >= 0:
            mm[input_ids == video_token_id] = 2

        batch["mm_token_type_ids"] = mm

        assert batch["attention_mask"].shape == input_ids.shape, (
            batch["attention_mask"].shape,
            input_ids.shape,
        )
        assert batch["labels"].shape == input_ids.shape, (
            batch["labels"].shape,
            input_ids.shape,
        )
        assert batch["mm_token_type_ids"].shape == input_ids.shape, (
            batch["mm_token_type_ids"].shape,
            input_ids.shape,
        )

        return batch

    trainer.data_collator = fixed_data_collator

    batch = next(iter(trainer.get_train_dataloader()))

    print("batch keys:", batch.keys())
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("labels:", batch["labels"].shape)
    print("mm_token_type_ids:", batch["mm_token_type_ids"].shape)

    print("Start training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    last_model_dir = Path(args.output_dir) / "last_model"
    print("Saving last model to:", last_model_dir)
    trainer.save_model(str(last_model_dir))
    processor.save_pretrained(str(last_model_dir))

    print("Training finished.")
    print("Output dir:", args.output_dir)

    if args.run_region_eval:
        print("Releasing training model before region eval sweep...", flush=True)
        del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        best = run_region_eval_sweep(args)
        if best is not None:
            print("[DONE] Best region checkpoint:", best["checkpoint"])
            print("[DONE] Best score:", best["score"])
        else:
            print("[DONE] Region eval skipped: no checkpoint found.")

    print("Done:", args.output_dir)

if __name__ == "__main__":
    main()