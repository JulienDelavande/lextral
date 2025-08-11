# Crafting the model
PYTHON = python
DATA_DIR = data/jsonl
RESULTS_DIR = data/evaluations
TRAIN_FILE = $(DATA_DIR)/ledgar_train_text.jsonl
VAL_FILE = $(DATA_DIR)/ledgar_validation_text.jsonl
TEST_FILE = $(DATA_DIR)/ledgar_test_text.jsonl
TRAIN_ID = be234d21-2d05-485b-9fb7-8809c026dbf9
VAL_ID = 2bcdfb1d-7e78-4b02-8cb1-d4766f895ba8
JOB_ID = 356ec036-1dd1-45cd-9896-71eae5180d0b
JOB_ID_SMALL = d14c01f7-4cb8-4acc-aed3-897d9c0afa05
BASE_MODEL = ministral-3b-latest
CLASSIFIER_MODEL = mistral-small-latest
MAX_SAMPLES = 1000

# Building and Deploying the app
DOCKER_REPO=juliendelavande
IMAGE_NAME=lextral
TAG=0.6
NAMESPACE=lextral
PG_IMAGE_REPO=$(DOCKER_REPO)/postgresql-pgvector
PG_TAG=17

.PHONY: install build_dataset train evaluate clean upload create-finetune-job start-finetune status build push deploy

install:
	@echo "Installing dependencies..."
	cd /backend && \ uv venv && \ uv sync
	cd experiments && \ uv venv && \ uv sync

## DATASET GENERATION ##
build_dataset:
	@echo "Generating JSONL datasets..."
	$(PYTHON) experiments/build_dataset_jsonl.py --split train --output_folder $(DATA_DIR) --max_item -1
	$(PYTHON) experiments/build_dataset_jsonl.py --split validation --output_folder $(DATA_DIR) --max_item -1
	$(PYTHON) experiments/build_dataset_jsonl.py --split test --output_folder $(DATA_DIR) --max_item -1

upload:
	@echo "Uploading datasets to Mistral cloud..."
	$(PYTHON) experiments/finetune_mistral.py upload --train_file $(TRAIN_FILE) --val_file $(VAL_FILE)


## TRAINING ##
create-finetune-job:
	@echo "Creating fine-tune job..."
	$(PYTHON) experiments/finetune_mistral.py create \
		--train_id $(TRAIN_ID) \
		--val_id $(VAL_ID)

start-finetune:
	@echo "Starting fine-tune job..."
	$(PYTHON) experiments/finetune_mistral.py start \
		--job_id $(JOB_ID)

status:
	@echo "Checking fine-tune job status..."
	$(PYTHON) experiments/finetune_mistral.py status --job_id $(JOB_ID)


## EVALUATION ##
evaluate:
	@echo "Evaluating the model..."
	$(PYTHON) experiments/evaluate.py --model_name $(BASE_MODEL) --model_strategy chatprompt --max_samples $(MAX_SAMPLES)

evaluate-async:
	@echo "Evaluating the model..."
	$(PYTHON) experiments/evaluate_chatprompt_async.py --model_name $(BASE_MODEL) --max_samples $(MAX_SAMPLES)

evaluate-baseline:
	@echo "Evaluating the model..."
	$(PYTHON) experiments/evaluate.py --model_name $(BASE_MODEL) --model_strategy chatprompt --max_samples $(MAX_SAMPLES)

evaluate-rag:
	@echo "Evaluating the model..."
	$(PYTHON) experiments/evaluate.py --model_name $(BASE_MODEL) --model_strategy rag --max_samples $(MAX_SAMPLES)

evaluate-classifier:
	@echo "Evaluating the model..."
	$(PYTHON) experiments/evaluate.py --model_name $(BASE_MODEL) --model_strategy classifier --max_samples $(MAX_SAMPLES)


## DOCKER BUILD AND DEPLOYMENT ##
build:
	docker buildx create --use --name mybuilder || true
	docker buildx build --platform linux/amd64 -t $(DOCKER_REPO)/$(IMAGE_NAME):$(TAG) ./backend --load

build-pg:
	@echo ">> Building PG+pgvector image $(PG_IMAGE_REPO):$(PG_TAG)"
	docker buildx create --use --name mybuilder || true
	docker buildx build --platform linux/amd64 -t $(PG_IMAGE_REPO):$(PG_TAG) -f backend/Dockerfile_pgvector backend --load

push-pg:
	docker push $(PG_IMAGE_REPO):$(PG_TAG)

push:
	docker push $(DOCKER_REPO)/$(IMAGE_NAME):$(TAG)

deploy:
	kubectl create namespace $(NAMESPACE) || true
	helm upgrade --install lextral ./kubernetes --namespace $(NAMESPACE) --set image.repository=$(DOCKER_REPO)/$(IMAGE_NAME) --set image.tag=$(TAG) \
		--set postgresql.image.repository=$(PG_IMAGE_REPO) --set postgresql.image.tag=$(PG_TAG)
