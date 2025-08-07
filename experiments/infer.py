import requests
import os
from mistral_utils import format_prompt, parse_response, label_name_to_id

MISTRAL_API_URL = os.getenv("MISTRAL_API_URL")  # e.g., https://api.mistral.ai/v1/chat/completions
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL")  # e.g. ft:gpt-7b:your-org/your-ft
BASE_MODEL = os.getenv("BASE_MODEL", "mistral-small")

def predict_label(text, model_type="finetuned"):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    if model_type == "finetuned":
        payload = {
            "model": FINE_TUNED_MODEL,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
        }
    else:  # prompting
        prompt = format_prompt(text)
        payload = {
            "model": BASE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }

    r = requests.post(MISTRAL_API_URL, json=payload, headers=headers)
    r.raise_for_status()
    completion = r.json()["choices"][0]["message"]["content"]

    return parse_response(completion)
