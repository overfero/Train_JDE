import os
from huggingface_hub import snapshot_download

repo_id = "sshao0516/CrowdHuman"
local_dir = os.path.join(os.getcwd(), "CrowdHuman")

print(f"Downloading {repo_id} to {local_dir}...")
path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)

print("Dataset downloaded to:", path)
