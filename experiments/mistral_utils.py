from datasets import load_dataset

ds = load_dataset("lex_glue", "ledgar", split='train')
label_names = ds.features["label"].names
label_id_to_name = {int(i): name.lower() for i, name in enumerate(label_names)}
label_name_to_id = {name.lower(): int(i) for i, name in enumerate(label_names)}

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
