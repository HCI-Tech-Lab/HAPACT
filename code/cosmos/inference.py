import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from eval import DATA_DIR, OUTPUT_DIR, align_special_tokens, extract_region_names_from_prompt, load_region_mapping, parse_regions_from_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--model_name", default="nvidia/Cosmos-Reason2-8B")
    parser.add_argument("--resume", default=str(OUTPUT_DIR / "best_region_checkpoint"))
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    train_jsonl = DATA_DIR / "spatial" / "train.jsonl"
    with train_jsonl.open("r", encoding="utf-8") as f:
        first_row = json.loads(next(f))
    region_names = extract_region_names_from_prompt(first_row["prompt"])
    if not region_names:
        raise ValueError("No region choices found in the training benchmark prompt.")

    description_to_id = load_region_mapping(DATA_DIR / "smplx_part_segmentation.json")
    if {name.lower() for name in region_names} != set(description_to_id):
        raise ValueError("Prompt regions do not match segmentation descriptions.")

    prompt = (
        "You are observing a movie frame. "
        f'The impact query is "{args.query}". '
        "Select all body regions physically affected by the impact in this frame. "
        f"The possible body regions are: {', '.join(region_names)}.\n"
        "Answer with only the selected body-region labels, separated by commas."
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": prompt},
    ]}]

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
    pred_regions = [description_to_id[name.lower()] for name in parse_regions_from_output(raw_output, region_names)]

    print("\nRaw Output:")
    print(raw_output)
    print("\nPredicted Regions:")
    print(pred_regions)


if __name__ == "__main__":
    main()
