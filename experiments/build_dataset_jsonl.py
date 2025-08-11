from datasets import load_dataset
import json
import argparse

OUTPUT_FOLDER='./data/jsonl'

def convert_ledgar_split(split: str, output_folder: str, max_items: int = -1):
    """
    Convert a specific split of the LEDGAR dataset to JSONL format.
    Args:
        split (str): The dataset split to convert (train, validation, or test).
        output_folder (str): The folder where the JSONL file will be saved.
        max_items (int): Maximum number of items to include in the output. Default is -1 (all items).
"""
    ds = load_dataset("lex_glue", "ledgar", split=split)
    
    label_names = ds.features["label"].names
    label_map = {str(i): name for i, name in enumerate(label_names)}

    output_file_name = f"ledgar_{split}_text.jsonl"
    output_path = f'{output_folder}/{output_file_name}'
    with open(output_path, "w") as f:
        for i, example in enumerate(ds):
            clause = example["text"]
            label_id = example["label"]
            label_str = label_map[str(label_id)]

            item = {
                "text": clause,
                "labels": {"clause-type": label_str}
            }
            f.write(json.dumps(item) + "\n")
            if i == max_items:
                break
    print(f'Dataset {split} succesfully generated and saved at {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LEDGAR dataset split to JSONL format.")
    parser.add_argument("--split", type=str, choices=["train", "validation", "test"], default="train",
                        help="Dataset split to convert (train, validation, or test).")
    parser.add_argument("--output_folder", type=str, default=OUTPUT_FOLDER, help="Output path for the JSONL file.")
    parser.add_argument("--max_item", type=int, default=10)
    args = parser.parse_args()
    convert_ledgar_split(args.split, args.output_folder, args.max_item)
