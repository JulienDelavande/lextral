from datasets import load_dataset
import json
from typing import List
from mistralai.models import UserMessage, SystemMessage

label_id_to_name = json.load(open("./labels/label_id_to_name.json"))
label_name_to_id = json.load(open("./labels/label_name_to_id.json"))

def format_prompt(text):
    labels_str = "\n".join([f"{i}: {name}" for i, name in label_id_to_name.items()])
    return f"""You are a contract clause classifier. You must classify the following clause into one of the following categories:
{labels_str}

Clause:
\"\"\"{text}\"\"\"

Respond with only the category name, exactly as written above.
"""

def parse_response(response):
    """
    Extract the label name and convert to ID.
    If ambiguous, fallback or return -1.
    """
    print(f"Parsing response: \n<<<{response}>>>\n")
    #print(f"Label map: {label_id_to_name}")
    for name in label_id_to_name.values():
        if name.lower() in response.lower():
            return label_name_to_id[name.lower()]
    print(f"Warning: could not parse response: {response}")
    return -1

def parse_response_name(response):
    """
    Extract the label name.
    If ambiguous, fallback or return -1.
    """
    print(f"Parsing response: \n<<<{response}>>>\n")
    #print(f"Label map: {label_id_to_name}")
    for name in label_id_to_name.values():
        if name.lower() in response.lower():
            return name.lower()
    print(f"Warning: could not parse response: {response}")
    return -1

def build_fewshot_prompt(query: str, neighbors: list[tuple[str, str, float]]) -> List:
    """
    neighbors: [(label_text, text, sim), ...]
    Construit un prompt few-shot: {clause -> label} * K, puis la clause cible.
    """
    shots = []
    for (label_text, clause_text, sim) in neighbors:
        shots.append(f"Clause:\n{clause_text}\nLabel:\n{label_text}\n")
    shots_block = "\n---\n".join(shots)
    user = (
        "You are given several examples of clauses with their labels.\n"
        "Infer the correct label for the final clause. Answer with the label only.\n\n"
        f"Examples:\n{shots_block}\n\n"
        f"Final clause:\n{query}\n"
        "Label:"
    )
    labels_str = "\n".join([f"{i}: {name}" for i, name in label_id_to_name.items()])
    system = f"You classify contract clauses into one of 100 predefined labels. Reply with the label only. labels: {labels_str}"
    return [SystemMessage(content=system),
            UserMessage(content=user)]