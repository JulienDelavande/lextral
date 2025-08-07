import argparse
import json
from datasets import load_dataset
from sklearn.metrics import classification_report
from infer import predict_label
from mistral_utils import label_id_to_name, name_to_label_id

def evaluate(model_type, max_samples=None):
    dataset = load_dataset("lex_glue", "ledgar", split="test")
    y_true, y_pred = [], []

    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        text = example["text"]
        true_label = example["label"]
        pred_label = predict_label(text, model_type=model_type)

        y_true.append(true_label)
        y_pred.append(pred_label)

        if i % 20 == 0:
            print(f"[{i}] True: {true_label}, Pred: {pred_label}")

    report = classification_report(y_true, y_pred, zero_division=0, digits=3)
    print(report)

    with open(f"results_{model_type}.json", "w") as f:
        json.dump({"y_true": y_true, "y_pred": y_pred}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["finetuned", "prompt"], required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    evaluate(args.model_type, max_samples=args.max_samples)
