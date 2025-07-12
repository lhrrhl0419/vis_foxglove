# vis_pyvista

## Installation

* pip install -e .

## Usage

Example Usage:

```bash
# server
python src/utils/vis_save.py # the scene construction can be found in self_test
# local
IP=<IP address> PORT=<port number> REMOTE_USER=<remote server's username> KEY=<path to ssh key> REMOTE_PATH=<directory vis's path on remote server> LOCAL_PATH=./tmp/vis python src/utils/vis_pyvista.py
```

The visualization of various elements in vis_save returns a list. These lists can be combined to create complex scenes, and the final saved result is a list of lists, each representing a moment in time. Currently, saving will result in a folder containing scene.json and a bunch of meshes, with the same mesh being saved only once.

vis_save can run on a remote server. After saving the folder, you can run vis_pyvista locally. By setting the remote server's path through environment variables (see src/utils/download.py), the content from the remote server will be automatically downloaded. You can then freely adjust the view with the mouse and use the progress bar in the upper right corner to switch between time steps. 

Additionally, there are some keyboard shortcuts, including:

- (number) r: re-download and read
- (number) o: re-read
- (number) d: re-download
- (number) s: save the current scene to another directory
- j, k: next frame and previous frame, respectively
- space: start/pause playback
- z: reset time to 0
- q: clear the number buffer

The (number) represents the scene number that can be input before the corresponding command. This number can specify the download/read/save location. If no number is set, the current scene number is used (initialized to 0).
