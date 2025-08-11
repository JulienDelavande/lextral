import json

label_id_to_name = json.load(open("./labels/label_id_to_name.json"))
label_name_to_id = json.load(open("./labels/label_name_to_id.json"))

def format_prompt(text):
    """
    Format the input text into a prompt for the model.
    """
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
    for name in label_id_to_name.values():
        if name.lower() in response.lower():
            return label_name_to_id[name.lower()]
    print(f"Warning: could not parse response: {response}")
    return -1
