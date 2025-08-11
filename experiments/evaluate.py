import argparse
import json
import os
from datasets import load_dataset
from sklearn.metrics import classification_report
from infer import predict_label

RESULT_FOLDER = './data/evaluations'
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL")

def save_results(y_true, y_pred, model_name, model_strategy, max_samples):
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    path = f"{RESULT_FOLDER}/{model_name}_{model_strategy}_{max_samples}bisbis.json"
    with open(path, "w") as f:
        json.dump({"y_true": y_true, "y_pred": y_pred}, f, indent=4)

def evaluate(model_name, model_strategy, max_samples=None, batch_size=1):
    dataset = load_dataset("lex_glue", "ledgar", split="test")
    y_true, y_pred = [], []
    if max_samples:
        #dataset = dataset.select(range(407, max_samples))
        dataset = dataset.select(range(0, max_samples))

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i: i + batch_size]
        print(batch)
        texts = [text for text in batch["text"]]
        labels = [label for label in batch["label"]]

        preds = predict_label(
            texts=texts,
            model_name=model_name,
            model_strategy=model_strategy
        )
        print(f"[{i}] True: {labels}, Pred: {preds}")

        y_true.extend(labels)
        y_pred.extend(preds)

        # Sauvegarder les résultats à chaque batch traité
        save_results(y_true, y_pred, model_name, model_strategy, max_samples)

        for j, (true, pred) in enumerate(zip(labels, preds)):
            print(f"[{i + j}] True: {true}, Pred: {pred}")

    report = classification_report(y_true, y_pred, zero_division=0, digits=3)
    print(report)

    # Sauvegarder le rapport final après traitement de tous les batches
    save_results(y_true, y_pred, model_name, model_strategy, max_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=[FINE_TUNED_MODEL, "mistral-small-latest", "ministral-3b-latest", "mistral-medium-latest"], default="mistral-small-latest", help="Model name to use for evaluation")
    parser.add_argument("--model_strategy", choices=["chatprompt", "classifier", "rag"], default="chatprompt")
    parser.add_argument("--max_samples", type=int, default=5)
    args = parser.parse_args()

    evaluate(model_name=args.model_name, model_strategy=args.model_strategy, max_samples=args.max_samples)
