from datasets import load_dataset
import json
import argparse

def convert_ledgar_split(split: str, output_path: str):
    ds = load_dataset("lex_glue", "ledgar", split=split)
    
    label_names = ds["train"].features["label"].names
    label_map = {str(i): name for i, name in enumerate(label_names)}

    with open(output_path, "w") as f:
        for example in ds:
            clause = example["text"]
            label_id = example["label"]
            label_str = label_map[str(label_id)]

            item = {
                "messages": [{"role": "user", "content": clause}],
                "labels": {"clause-type": label_str}
            }
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LEDGAR dataset split to JSONL format.")
    parser.add_argument("--split", type=str, choices=["train", "validation", "test"], required=True,
                        help="Dataset split to convert (train, validation, or test).")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the JSONL file.")
    args = parser.parse_args()
    convert_ledgar_split(args.split, args.output_path)
