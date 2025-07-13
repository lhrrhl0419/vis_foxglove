import os
import subprocess
from tqdm import tqdm


def download_with_rsync(
    host=None,
    port=None,
    username=None,
    private_key_path=None,
    remote_dir=None,
    local_dir=None,
):
    host = os.environ.get("IP") if host is None else host
    port = os.environ.get("PORT") if port is None else port
    username = os.environ.get("REMOTE_USER") if username is None else username
    private_key_path = (
        os.environ.get("KEY") if private_key_path is None else private_key_path
    )
    remote_dir = os.environ.get("REMOTE_PATH") if remote_dir is None else remote_dir
    local_dir = os.environ.get("LOCAL_PATH") if local_dir is None else local_dir
    os.makedirs(local_dir, exist_ok=True)

    # Construct the rsync command
    cmd = [
        "rsync",
        "-Pavz",  # Archive mode, compress, show progress
        "-e",
        f"ssh -p {port} -i {private_key_path}",
        f"{username}@{host}:{os.path.join(remote_dir, '*')}",
        local_dir,
    ]
    print(" ".join(cmd))

    # Run rsync with real-time progress output
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as proc:
        for line in tqdm(proc.stdout, desc="Downloading", unit="files"):
            print(line.strip())  # Display rsync output (optional)


# Example Usage
if __name__ == "__main__":
    download_with_rsync()
