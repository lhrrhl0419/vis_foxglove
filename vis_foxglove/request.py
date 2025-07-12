import socket
import socketserver

from vis_foxglove.ports import REQUEST_PORT
from vis_foxglove.download import download_with_rsync

def request_sync(local_machine_ip="127.0.0.1", port=REQUEST_PORT) -> bool:
    """Send a sync request to the local machine"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((local_machine_ip, port))
            s.sendall(b"SYNC_REQUEST")
            response = s.recv(1024).decode('utf-8')
            
            if response == "SYNC_COMPLETE":
                print("Sync completed successfully")
                return True
            else:
                print("Sync failed on local machine")
    except Exception as e:
        print(f"Failed to send sync request: {e}")
    return False


class SyncRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        data = self.request.recv(1024).decode('utf-8').strip()
        if data == "SYNC_REQUEST":
            print("Received sync request from server")
            try:
                download_with_rsync()
                self.request.sendall(b"SYNC_COMPLETE")
            except Exception as e:
                print(f"Sync failed: {e}")
                self.request.sendall(b"SYNC_FAILED")


def start_listener(port=REQUEST_PORT):
    with socketserver.TCPServer(("0.0.0.0", port), SyncRequestHandler) as server:
        print(f"Local sync listener running on port {port}")
        server.serve_forever()


if __name__ == "__main__":
    request_sync()