import os
import requests
import json
from mistralai import Mistral
from mistralai.models import UserMessage
from mistral_utils import format_prompt, parse_response

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL", "mistral-small")
BASE_MODEL = os.getenv("BASE_MODEL", "mistral-small")
URL_PREDICT_RAGMODEL = os.getenv("URL_PREDICT_RAGMODEL", "https://lextral.delavande.fr/predict_rag")
URL_PREDICT_RAG = os.getenv("URL_PREDICT_RAG", "https://lextral.delavande.fr/predict_rag_knn")

client = Mistral(api_key=MISTRAL_API_KEY)

def predict_label(texts, model_name=BASE_MODEL, model_strategy="prompt"):
    if model_strategy == "classifier":
        resp = client.classifiers.classify(
            model=model_name,
            inputs=texts,
        )
        preds = []
        for item in resp.results:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif not isinstance(item, dict):
                try:
                    item = json.loads(item.json())
                except Exception:
                    item = dict(item.__dict__)
            try:
                _, payload = next(
                    (k, v) for k, v in item.items()
                    if isinstance(v, dict) and "scores" in v
                )
            except StopIteration:
                raise ValueError(f"No key found: {item}")

            scores = payload["scores"]  # dict {label_name: score}
            if not isinstance(scores, dict) or not scores:
                raise ValueError(f"'scores' is empty or invalid in the result: {item}")

            best_label_name = max(scores.items(), key=lambda kv: kv[1])[0]
            preds.append(parse_response(best_label_name))
        return preds
    
    elif model_strategy == "chatprompt":
        prompts = [format_prompt(text) for text in texts]
        messages = [UserMessage(content=prompt) for prompt in prompts]
        responses = [client.chat.complete(
            model=model_name,
            messages=[message],
            temperature=0,
            ) for message in messages]
        return [parse_response(response.choices[0].message.content) for response in responses]
    
    elif model_strategy == "rag+model":
        url = URL_PREDICT_RAGMODEL

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        data = {
            "texts": texts,
            "top_k": 5,
            "min_sim": 0
        }

        response = requests.post(url, headers=headers, json=data)
        return [parse_response(predicted_label) for predicted_label in response.json().get("predicted_classes", [])]
    
    elif model_strategy == "rag":
        url = URL_PREDICT_RAG
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        data = {
            "texts": texts,
            "top_k": 1,
            "min_sim": 0,
            "strategy": "weighted",
            "power": 2,
            "tau": 0.1,
            "abstain_min_conf": 0
        }

        response = requests.post(url, headers=headers, json=data)
        return [parse_response(predicted_label) for predicted_label in response.json().get("predicted_classes", [])]
    
    else:
        raise ValueError(f"Unknown strategy: {model_strategy}")


async def predict_label_async(text, model_name=BASE_MODEL):
    prompt = format_prompt(text)
    messages = [UserMessage(content=prompt)]
    response = await client.chat.complete_async(
        model=model_name,
        messages=messages,
        temperature=0,
    )

    return parse_response(response.choices[0].message.content)
