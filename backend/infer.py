import os
import json
from mistralai import Mistral
from mistralai.models import UserMessage
from mistral_utils import format_prompt, parse_response_name

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL") or "mistral-small"
BASE_MODEL = os.getenv("BASE_MODEL", "mistral-small")

client = Mistral(api_key=MISTRAL_API_KEY)

def predict_label(texts, model_name=BASE_MODEL, model_strategy="chatprompt"):
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
            preds.append(parse_response_name(best_label_name))
        return preds, texts

    elif model_strategy == "chatprompt":
        prompts = [format_prompt(text) for text in texts]
        messages = [UserMessage(content=prompt) for prompt in prompts]
        responses = [client.chat.complete(
            model=model_name,
            messages=[message],
            temperature=0,
            ) for message in messages]
        return [parse_response_name(response.choices[0].message.content) for response in responses], messages
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

    return parse_response_name(response.choices[0].message.content), messages

def chat_complete(messages):
    resp = client.chat.complete(
                model=BASE_MODEL,
                messages=messages,
                temperature=0,
            )
    pred = parse_response_name(resp.choices[0].message.content)
    return pred