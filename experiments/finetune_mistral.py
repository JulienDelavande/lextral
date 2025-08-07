import os
from mistralai.client import Mistral

def upload_file(client, path):
    with open(path, "rb") as f:
        file = client.files.upload(file={"file_name": os.path.basename(path), "content": f})
    print(f"Uploaded {path} → ID = {file.id}")
    return file

def create_ft_job(client, training_file_id, validation_file_id):
    job = client.fine_tuning.jobs.create(
        model="ministral-3b-latest",
        job_type="classifier",
        training_files=[{"file_id": training_file_id, "weight": 1}],
        validation_files=[validation_file_id],
        hyperparameters={
            "training_steps": 100,         # Ajuste selon dataset
            "learning_rate": 0.0001
        },
        auto_start=False,
        # integrations=[
        #     {
        #         "project": "ClauseFT",
        #         "api_key": os.environ["WANDB_API_KEY"]
        #     }
        # ]
    )
    print(f"Created job: {job.id}")
    return job

def launch_finetuning():
    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)

    training = upload_file(client, "training_file.jsonl")
    validation = upload_file(client, "validation_file.jsonl")

    job = create_ft_job(client, training.id, validation.id)

    input("Job created. Press Enter to launch fine-tuning...")
    client.fine_tuning.jobs.start(job.id)
    print(f"Job {job.id} started!")

if __name__ == "__main__":
    launch_finetuning()
