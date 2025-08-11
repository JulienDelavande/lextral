import datetime
import json
import os
import argparse
from mistralai import Mistral
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
WANDB_KEY = os.getenv("WANDB_KEY", None)
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

def upload_file(path):
    with open(path, "rb") as f:
        file = client.files.upload(file={"file_name": os.path.basename(path), "content": f})
    print(f"Uploaded {path} : ID = {file.id}")
    return file

def create_job(train_id, val_id, base_model, steps=100, lr=0.0001, wandb=True):
    integrations = []
    if wandb and WANDB_KEY:
        integrations.append({
            "project": "finetuning",
            "api_key": WANDB_KEY
        })
    job = client.fine_tuning.jobs.create(
        model=base_model,
        job_type="classifier",
        training_files=[{"file_id": train_id, "weight": 1}],
        validation_files=[val_id],
        hyperparameters={
            "training_steps": steps,
            "learning_rate": lr
        },
        auto_start=False,
        integrations=integrations
    )
    print(f"Created job: {job.id}")
    print(f"Job details:\n{job}")
    return job

def start_job(job_id):
    print(f"Starting job {job_id}...")
    job = client.fine_tuning.jobs.start(job_id=job_id)
    print(f"Job {job_id} started.")
    print(f"Job details:\n{job}")

def get_status(job_id):
    job = client.fine_tuning.jobs.get(job_id=job_id)
    estimated_start_ts = job.metadata.estimated_start_time
    print(f"Status for job {job.id}:\n"
          f"- Model: {job.model}\n"
          f"- Status: {job.status}\n"
          f"- Fine-tuned Model: {job.fine_tuned_model}\n"
          f"- Integrations: {job.integrations}\n")
    if estimated_start_ts:
        estimated_start_dt = datetime.fromtimestamp(estimated_start_ts)
        print(f"- Estimated start time: {estimated_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("- Estimated start time: N/A")

def list_jobs():
    jobs = client.fine_tuning.jobs.list()
    print("📝 Existing fine-tuning jobs:")
    for job in jobs:
        print(f"- {job.id}: {job.status}")

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # upload
    upload_p = subparsers.add_parser("upload")
    upload_p.add_argument("--train_file", required=True)
    upload_p.add_argument("--val_file", required=True)

    # create
    create_p = subparsers.add_parser("create")
    create_p.add_argument("--train_id", required=True)
    create_p.add_argument("--val_id", required=True)
    create_p.add_argument("--base_model", default="ministral-3b-latest")
    create_p.add_argument("--steps", type=int, default=100)
    create_p.add_argument("--lr", type=float, default=0.0001)

    # start
    start_p = subparsers.add_parser("start")
    start_p.add_argument("--job_id", required=True)

    # status
    status_p = subparsers.add_parser("status")
    status_p.add_argument("--job_id", required=True)

    # list
    list_p = subparsers.add_parser("list")

    args = parser.parse_args()

    if args.command == "upload":
        train = upload_file(args.train_file)
        val = upload_file(args.val_file)
        print(f"🏷️ Train ID: {train.id}\n🏷️ Val ID: {val.id}")
    elif args.command == "create":
        job = create_job(args.train_id, args.val_id, args.base_model, args.steps, args.lr)
        print(f"Job created with ID: {job.id}")
        print(f"Fine-tuned model will be available at: {job.fine_tuned_model}")
    elif args.command == "start":
        start_job(args.job_id)
    elif args.command == "status":
        get_status(args.job_id)
    elif args.command == "list":
        list_jobs()
