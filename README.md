# Lextral

Lextral is a project designed to develop a model for classifying contractual clauses using the [LexGLUE/LEDGAR](https://huggingface.co/datasets/lex_glue) dataset.

Try it out online at [https://lextral.delavande.fr](https://lextral.delavande.fr).

We explored three main strategies for contract clause classification:

1. **Prompt-based classification** – Use a general-purpose LLM (e.g., *mistral-small*) with a simple zero-shot prompt listing all possible categories. The model directly returns the predicted category name.
2. **Retrieval-augmented classification** – Embed all training clauses (60k) using *Mistral Embed*, store them in a `pgvector` database with an HNSW index, and at inference time embed the query clause, retrieve the *k* most similar clauses and their labels, then include them as few-shot examples in the prompt.
3. **Fine-tuned classification** – Fine-tune a larger model (*Ministral 3B*) with a classification head on the 60k training clauses using LoRA, enabling the model to directly predict the correct label without additional context retrieval.

The project is organized into two main components:

* **Experiments** – for running experiments and fine-tuning the model.
* **Backend** – for the deployed version of the solution, available online at [https://lextral.delavande.fr](https://lextral.delavande.fr) and the api documentation at [https://lextral.delavande.fr/docs](https://lextral.delavande.fr/docs).

---

## Installing Dependencies

To install all required dependencies, run:

```bash
cd backend
uv sync
source backend/venv/bin/activate
```

This will create a virtual environment and synchronize the dependencies listed in `pyproject.toml`.

To set the environment variables, you can create a `secrets.env` file with the following content:

```bash
MISTRAL_API_KEY="your_mistral_api_key_here"
WANDB_KEY="your_wandb_api_key_here" # This is optional, only needed for Weights & Biases integration
DB_PASSWORD="your_db_password_here" # This is the password for the PostgreSQL database
```

then load it in your shell:

```bash
set -a
source secrets.env
source .env
set +a
```

---

## Dataset Preparation

To generate the JSONL files for training, validation, and testing, run:

```bash
make build_dataset
```

The generated files will be stored in the `data/jsonl` directory.

---

## Fine-tuning the Mistral Model

### Step 1 – Upload the datasets

Upload the datasets to Mistral Cloud:

```bash
make upload
```

### Step 2 – Create a fine-tuning job

Create a fine-tuning job:

```bash
make create-finetune-job
```

### Step 3 – Start the fine-tuning

Start the fine-tuning job:

```bash
make start-finetune
```

### Step 4 – Check the job status

Monitor the fine-tuning progress:

```bash
make status
```

## Self tuned you model

```bash
# For the Head only
python experiments/finetune_mistral_self.py --base_model mistralai/Ministral-8B-Instruct-2410 --output_dir ./outputs_ministral8b_head

# For the Head + LoRA
python experiments/finetune_mistral_self.py --base_model mistralai/Ministral-8B-Instruct-2410 --output_dir ./outputs_ministral8b_headlora --lora

# For evaluation
python experiments/evaluate_mistral_self.py --ckpt_dir ./outputs_ministral8b_head
```


---

## Running Experiments

### Synchronous evaluation

Run a synchronous evaluation:

```bash
make evaluate
make evaluate-baseline # This runs the baseline evaluation a base model
make evaluate-rag # This runs the strategy with of classifying clauses using RAG (You have to run the database provisioning first)
make evaluate-classifier # This runs the classifier evaluation (fine-tuned model on the LexGLUE/LEDGAR dataset)
```

### Asynchronous evaluation

Run a large-scale asynchronous evaluation:

```bash
make evaluate-async
```

Evaluation results will be saved in the `data/evaluations` directory.


---

## Provisioning the Database Locally

For evaluation of the RAG (Retrieval-Augmented Generation) model, a local PostgreSQL database with pgvector is required. Follow these steps to provision the database:

### Step 1 – Start the PostgreSQL database

Run the initialization script to set up the database:

```bash
bash db/init.sh
```

### Step 2 – Apply database migrations

Apply the necessary migrations to create the required tables:

```bash
bash db/apply_migrations.py
```

### Step 3 – Build embeddings

Generate and store embeddings in the database:

```bash
python db/scripts/build_embeddings_mistral.py
```

Ensure the database is running and accessible before proceeding with the evaluation.


---

## Deploying the Backend

### Prerequisites

Ensure you have Docker and Kubernetes installed on your machine.
Change the var in the `Makefile` to match your docker hub repository `DOCKER_REPO`.

### Building Docker images

Build the backend Docker image:

```bash
make build
```

Build the PostgreSQL image with pgvector:

```bash
make build-pg
```

### Pushing Docker images

Push the images to the container registry:

```bash
make push
make push-pg
```

### Deploying to Kubernetes

Deploy the solution to Kubernetes using Helm:

```bash
make deploy
```

This will create a `lextral` namespace and deploy both the application and the PostgreSQL database with pgvector.

### Provisioning the Database on Kubernetes

To connect to the PostgreSQL database running in Kubernetes, use the following commands:

```bash
kubectl -n lextral port-forward svc/lextral-postgresql 5433:5432
``` 

Then create the database:

```bash
createdb -h 127.0.0.1 -p 5433 -U app "lextral-db"
```
Apply the migrations:

```bash
export DB_PORT=5433
python db/apply_migrations.py
```
Then build the embeddings and store them in the database:

```bash
python db/scripts/build_embeddings_mistral.py
```

---

## References

* Dataset: [LexGLUE/LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue)
* Model: [Mistral](https://mistral.ai/fr)
* Deployment: [Helm](https://helm.sh/), [Kubernetes](https://kubernetes.io/)
