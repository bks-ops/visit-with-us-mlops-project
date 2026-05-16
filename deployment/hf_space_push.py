
from huggingface_hub import create_repo, upload_folder

repo_id = "bksharma/visit-with-us-space"

# Create space repo
create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True
)

# Upload deployment folder
upload_folder(
    repo_id=repo_id,
    folder_path="tourism_project/deployment",
    repo_type="space"
)

print("Deployment files pushed successfully.")
