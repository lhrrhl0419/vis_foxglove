import os
from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the local file server!"

@app.route('/<path:filename>')
def serve_file(filename):
    directory = os.path.abspath(os.getenv("LOCAL_PATH", "tmp/vis"))
    return send_from_directory(directory, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=19685)