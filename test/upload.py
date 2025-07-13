from pathlib import Path

from foxglove_data_platform.client import Client

token = "fox_sk_7XVjyWGvyzOjXe6ca7kHvNPfFJBrNeDP"
device_id = "dev_0dlSh93OkdrMFXax"

client = Client(token=token)

# Upload bytes
mcap_data = Path("quickstart-python.mcap").read_bytes()

client.upload_data(
    device_id=device_id,
    filename="test mcap upload",
    data=mcap_data,
    callback=lambda size, progress: print(size, progress),
)
