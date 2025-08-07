import re

# Optionnel : mapping label ID ↔ texte lisible
label_id_to_name = {
    0: "Confidentiality",
    1: "Termination",
    2: "Payments",
    # ...
}
name_to_label_id = {v.lower(): k for k, v in label_id_to_name.items()}

def format_prompt(text):
    return f"""You are a contract clause classifier. You must classify the following clause into one of the following categories:
{', '.join(label_id_to_name.values())}

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
            return name_to_label_id[name.lower()]
    print(f"Warning: could not parse response: {response}")
    return -1
