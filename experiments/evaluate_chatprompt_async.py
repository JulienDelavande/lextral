import argparse
import asyncio
import json
import os
from datasets import load_dataset
from sklearn.metrics import classification_report
from infer import predict_label_async

RESULT_FOLDER = "./data/evaluations"
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL", "mistral-small-latest")
BATCH_SIZE = 1000
JITTER = 1

async def process_batch(batch, model_name):
    """Process a batch of text asynchronously and return predictions.

    Args:
        batch (list): List of text inputs to process.
        model_name (str): The name of the model to use for predictions.

    Returns:
        list: List of predicted labels.
    """
    print(f"Processing batch of size {len(batch)} with model {model_name}")
    tasks = [
        predict_label_async(example, model_name=model_name)
        for example in batch
    ]
    return await asyncio.gather(*tasks)


async def evaluate_async(model_name, max_samples=None):
    """Evaluate the model asynchronously on the LEDGAR dataset.

    Args:
        model_name (str): The name of the model to use for evaluation.
        max_samples (int, optional): Maximum number of samples to evaluate. Defaults to None (all samples).
    """
    print(f"Evaluating model: {model_name} with max_samples={max_samples}")
    dataset = load_dataset("lex_glue", "ledgar", split="test")
    if max_samples:
        dataset = dataset.select(range(max_samples))

    y_true, y_pred = [], []

    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset["text"][i:i+BATCH_SIZE]
        predictions = await process_batch(batch, model_name)

        for j, label in enumerate(dataset["label"][i:i+BATCH_SIZE]):
            true_label = label
            pred_label = predictions[j]

            y_true.append(true_label)
            y_pred.append(pred_label)

            print(f"[{i+j}] True: {true_label}, Pred: {pred_label}")

        await asyncio.sleep(JITTER)

    report = classification_report(y_true, y_pred, zero_division=0, digits=3)
    print(report)

    os.makedirs(RESULT_FOLDER, exist_ok=True)
    path = f"{RESULT_FOLDER}/{model_name}_chatprompt_{max_samples}.json"
    with open(path, "w") as f:
        json.dump({"report": report, "y_true": y_true, "y_pred": y_pred}, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=[FINE_TUNED_MODEL, "mistral-small-latest", "ministral-3b-latest"], default="mistral-small-latest")
    parser.add_argument("--model_strategy", choices=["prompt"], default="prompt")
    parser.add_argument("--max_samples", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(evaluate_async(model_name=args.model_name, max_samples=args.max_samples))
