import os
from typing import Optional, List, Dict, Union
import numpy as np
import curses
import matplotlib.pyplot as plt

try:
    import torch
except ImportError:
    torch = None
from time import sleep
from transforms3d.quaternions import mat2quat
import matplotlib.colors as mcolors
import subprocess

import foxglove
from foxglove import Context
from foxglove.channels import (
    SceneUpdateChannel,
    PointCloudChannel,
    RawImageChannel,
    LogChannel,
    CompressedVideoChannel,
)
from foxglove.schemas import (
    SpherePrimitive,
    Pose,
    Vector3,
    Quaternion,
    Color,
    SceneUpdate,
    SceneEntity,
    ModelPrimitive,
    LinePrimitive,
    Point3,
    ArrowPrimitive,
    CubePrimitive,
    RawImage,
    Log,
    CompressedVideo,
    PointCloud,
    PackedElementField,
    PackedElementFieldNumericType
)

from vis_foxglove.utils import safe_copy, rm_r, to_numpy, gen_uuid, to_number
from vis_foxglove.pin_model import PinRobotModel

if torch is not None:
    Ary = Union[np.ndarray, torch.tensor]
else:
    Ary = np.ndarray

robot_models: Dict[str, PinRobotModel] = dict()


def get_video_codec(file_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return f"Error: {result.stderr.strip()}"


def convert_to_rel_path(path: str) -> str:
    if path[0] != "/":
        return path
    return f"absolute_path{path}"


def to_scene_entity_pc(scenes: List[Dict[str, list]]) -> SceneEntity:
    entity = SceneEntity(
        arrows=[arrow for scene in scenes for arrow in scene.get("arrows", [])],
        cubes=[cube for scene in scenes for cube in scene.get("cubes", [])],
        spheres=[sphere for scene in scenes for sphere in scene.get("spheres", [])],
        cylinders=[
            cylinder for scene in scenes for cylinder in scene.get("cylinders", [])
        ],
        lines=[line for scene in scenes for line in scene.get("lines", [])],
        triangles=[
            triangle for scene in scenes for triangle in scene.get("triangles", [])
        ],
        texts=[text for scene in scenes for text in scene.get("texts", [])],
        models=[model for scene in scenes for model in scene.get("models", [])],
    )
    pc_fields = [
        PackedElementField(name="x", offset=0, type=PackedElementFieldNumericType.Float32),
        PackedElementField(name="y", offset=4, type=PackedElementFieldNumericType.Float32),
        PackedElementField(name="z", offset=8, type=PackedElementFieldNumericType.Float32),
        PackedElementField(name="red", offset=12, type=PackedElementFieldNumericType.Uint8),
        PackedElementField(name="green", offset=13, type=PackedElementFieldNumericType.Uint8),
        PackedElementField(name="blue", offset=14, type=PackedElementFieldNumericType.Uint8),
        PackedElementField(name="alpha", offset=15, type=PackedElementFieldNumericType.Uint8),
    ]
    pc = []
    for scene in scenes:
        for pointcloud in scene.get("pointclouds", []):
            pc.append(pointcloud)
    if len(pc) == 0:
        return entity, None
    pc = np.concatenate(pc, axis=0)
    # Split xyz and rgba
    xyz = pc[:, :3].astype(np.float32)
    rgba = (pc[:, 3:7] * 255).astype(np.uint8)

    # Structured buffer: 16 bytes per point
    combined = np.zeros(len(pc), dtype=np.dtype([
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1"),
    ]))

    combined["x"] = xyz[:, 0]
    combined["y"] = xyz[:, 1]
    combined["z"] = xyz[:, 2]
    combined["r"] = rgba[:, 0]
    combined["g"] = rgba[:, 1]
    combined["b"] = rgba[:, 2]
    combined["a"] = rgba[:, 3]

    pc = PointCloud(
        point_stride=16,
        fields=pc_fields,
        data=combined.tobytes(),
    )
    return entity, pc



global_vis: "Vis" = None

class Vis:
    def __init__(self):
        self.ctx = Context()
        self.server = foxglove.start_server(context=self.ctx, port=int(os.environ.get("FOXGLOVE_PORT", 8765)))
    
    @staticmethod
    def get() -> "Vis":
        global global_vis
        if global_vis is None:
            global_vis = Vis()
        return global_vis

    @staticmethod
    def to_pose(trans: Ary, rot: Optional[Ary] = None) -> Pose:
        if rot is None:
            rot = np.eye(3)
        wxyz = mat2quat(rot)
        return Pose(
            position=Vector3(x=trans[0], y=trans[1], z=trans[2]),
            orientation=Quaternion(x=wxyz[1], y=wxyz[2], z=wxyz[3], w=wxyz[0]),
        )

    @staticmethod
    def to_color(color: Union[str, Ary] = None, opacity: float = 1.0, return_list: bool = False) -> Union[Color, List[float]]:
        if isinstance(color, str):
            color = mcolors.to_rgb(color)
        if color is None:
            color = np.array([1, 0, 0])
        if len(color) == 3:
            color = np.append(color, opacity)
        if return_list:
            return color
        return Color(r=color[0], g=color[1], b=color[2], a=color[3])

    @staticmethod
    def rand_color() -> Ary:
        return np.random.rand(3)

    @staticmethod
    def sphere(
        trans: Ary = None,
        radius: float = None,
        color: Union[str, Ary] = None,
        opacity: float = None,
    ) -> List[Dict[str, list]]:
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        radius = 0.1 if radius is None else to_number(radius)
        color = "blue" if color is None else color
        opacity = 1.0 if opacity is None else opacity
        return [
            dict(
                spheres=[
                    SpherePrimitive(
                        pose=Vis.to_pose(trans),
                        size=Vector3(x=radius * 2, y=radius * 2, z=radius * 2),
                        color=Vis.to_color(color, opacity),
                    )
                ]
            )
        ]
    
    @staticmethod
    def pc(
        points: Ary,
        color: Union[str, Ary] = None,
        value: Optional[Ary] = None,
        colormap: str = "viridis",
        normalize: bool = True,
        opacity: float = None,
    ) -> List[Dict[str, list]]:
        points = to_numpy(points)
        opacity = 1.0 if opacity is None else opacity
        if value is None:
            color = "blue" if color is None else color
            if isinstance(color, str) or len(color.shape) == 1:
                color = np.array(Vis.to_color(color, opacity, return_list=True))
                colors = color[None].repeat(len(points), axis=0)
            else:
                colors = color
                if colors.shape[1] == 3:
                    opacity = np.ones(len(points)) * opacity
                    colors = np.concatenate([colors, opacity], axis=1)
        else:
            value = to_numpy(value)
            if normalize:
                value = (value - value.min()) / (value.max() - value.min())
            colors = plt.cm.get_cmap(colormap)(value)
            colors[:, 3] *= opacity
        return [
            dict(
                pointclouds=[np.concatenate([points, colors], axis=1)]
            )
        ]

    @staticmethod
    def box(
        size: Ary,
        trans: Ary = None,
        rot: Ary = None,
        color: Union[str, Ary] = None,
        opacity: float = None,
    ) -> List[Dict[str, list]]:
        size = np.ones(3) if size is None else to_numpy(size)
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        rot = np.eye(3) if rot is None else to_numpy(rot)
        color = "blue" if color is None else color
        opacity = 1.0 if opacity is None else opacity
        return [
            dict(
                cubes=[
                    CubePrimitive(
                        pose=Vis.to_pose(trans, rot),
                        size=Vector3(x=size[0], y=size[1], z=size[2]),
                        color=Vis.to_color(color, opacity),
                    )
                ]
            )
        ]

    @staticmethod
    def pose(
        trans: Union[np.ndarray, torch.tensor],  # (3, )
        rot: Union[np.ndarray, torch.tensor],  # (3, 3)
        width: int = 0.01,
        length: float = 0.1,
        name: str = None,
        color: Union[str, Ary] = None,
    ) -> List[Dict[str, list]]:
        ret = []
        for i in range(3):
            if color is None:
                this_color = "red" if i == 0 else "green" if i == 1 else "blue"
            else:
                this_color = color
            ret.extend(
                Vis.arrow(
                    p1=trans,
                    direction=rot[:, i],
                    shaft_length=length,
                    shaft_diameter=width,
                    color=this_color,
                )
            )
        return ret
        
    @staticmethod
    def line(
        p1: Ary, p2: Ary, thickness: float = None, color: Union[str, Ary] = None
    ) -> List[Dict[str, list]]:
        p1 = to_numpy(p1)
        p2 = to_numpy(p2)
        thickness = 0.1 if thickness is None else to_number(thickness)
        color = "blue" if color is None else color
        return [
            dict(
                lines=[
                    LinePrimitive(
                        points=[
                            Point3(x=p1[0], y=p1[1], z=p1[2]),
                            Point3(x=p2[0], y=p2[1], z=p2[2]),
                        ],
                        thickness=thickness,
                        color=Vis.to_color(color),
                    )
                ]
            )
        ]

    @staticmethod
    def arrow(
        p1: Ary,
        direction: Ary,
        shaft_length: float = None,
        shaft_diameter: float = None,
        head_length: float = None,
        head_diameter: float = None,
        color: Union[str, Ary] = None,
    ) -> List[Dict[str, list]]:
        p1 = to_numpy(p1)
        direction = to_numpy(direction)
        shaft_length = 0.1 if shaft_length is None else to_number(shaft_length)
        shaft_diameter = 0.01 if shaft_diameter is None else to_number(shaft_diameter)
        head_length = shaft_length / 5 if head_length is None else to_number(head_length)
        head_diameter = shaft_diameter * 2 if head_diameter is None else to_number(head_diameter)
        color = "blue" if color is None else color
        rot = np.zeros((3, 3))
        direction = direction / np.linalg.norm(direction)
        rot[:, 0] = direction
        rot[np.abs(direction).argmin(), 1] = 1.0
        rot[:, 1] -= np.dot(rot[:, 1], rot[:, 0]) * rot[:, 0]
        rot[:, 1] /= np.linalg.norm(rot[:, 1])
        rot[:, 2] = np.cross(rot[:, 0], rot[:, 1])
        return [
            dict(
                arrows=[
                    ArrowPrimitive(
                        pose=Vis.to_pose(p1, rot),
                        shaft_length=shaft_length,
                        shaft_diameter=shaft_diameter,
                        head_length=head_length,
                        head_diameter=head_diameter,
                        color=Vis.to_color(color),
                    )
                ]
            )
        ]

    @staticmethod
    def mesh(
        path: str,
        trans: Ary = None,
        rot: Ary = None,
        scale: Union[float, Ary] = None,
        color: Optional[Union[str, Ary]] = None,
        opacity: float = None,
        overwrite_mesh_path: Optional[str] = None,
    ) -> List[Dict[str, list]]:
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        rot = np.eye(3) if rot is None else to_numpy(rot)
        if isinstance(scale, float):
            scale = np.array([scale, scale, scale])
        scale = np.ones(3) if scale is None else to_numpy(scale)
        color = "pink" if color is None else color
        opacity = 1.0 if opacity is None else opacity

        if path.startswith("http://"):
            url = path
            mesh_list = []
        else:
            if overwrite_mesh_path is not None:
                new_path = overwrite_mesh_path
            else:
                new_path = convert_to_rel_path(path)
            url = f"http://localhost:19685/{new_path}"
            mesh_list = [(path, new_path)]

        if path.endswith(".obj"):
            media_type = "model/obj"
        elif path.lower().endswith(".stl"):
            media_type = "model/stl"

        return [
            dict(
                models=[
                    ModelPrimitive(
                        pose=Vis.to_pose(trans, rot),
                        scale=Vector3(x=scale[0], y=scale[1], z=scale[2]),
                        url=url,
                        media_type=media_type,
                        color=Vis.to_color(color, opacity),
                        override_color=True,
                    )
                ],
                mesh_list=mesh_list,
            )
        ]

    @staticmethod
    def robot(
        urdf: str,
        qpos: Ary,
        trans: Ary = None,
        rot: Ary = None,
        opacity: float = None,
        color: Optional[Union[str, Ary]] = None,
        mesh_type: str = "visual",
        name: str = None,
    ):

        trans = np.zeros((3,)) if trans is None else to_numpy(trans).reshape(3)
        rot = np.eye(3) if rot is None else to_numpy(rot).reshape(3, 3)
        qpos = to_numpy(qpos).reshape(-1)
        color = "violet" if color is None else color
        opacity = 1.0 if opacity is None else opacity
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = trans
        name = gen_uuid() if name is None else name

        if urdf not in robot_models:
            robot_models[urdf] = PinRobotModel(urdf)

        poses = robot_models[urdf].fk_mesh(qpos, mode=mesh_type)
        lst = []
        for mesh_id, ((mesh_type, mesh_param), (mesh_trans, mesh_rot)) in enumerate(
            zip(robot_models[urdf].meshes[mesh_type], poses)
        ):
            if mesh_type == "sphere":
                lst += Vis.sphere(
                    trans=rot @ mesh_trans + trans,
                    radius=mesh_param["radius"],
                    opacity=opacity,
                    color=color,
                )
            elif mesh_type == "mesh":
                overwrite_mesh_path = mesh_param["path"].replace(
                    os.path.dirname(urdf), name
                )
                lst += Vis.mesh(
                    path=mesh_param["path"],
                    trans=rot @ mesh_trans + trans,
                    rot=rot @ mesh_rot,
                    opacity=opacity,
                    color=color,
                    overwrite_mesh_path=overwrite_mesh_path,
                )
        return lst

    def show(
        self,
        lst: List[List[Dict[str, list]]],
        path: Optional[str] = None,
        topic: str = "/scene",
        pc_topic: str = "/pointcloud",
        dt: float = 0.2,
        mesh_dir: str = "tmp/vis",
        non_blocking_t: Optional[int] = None,
    ):
        scene_channel = SceneUpdateChannel(topic, context=self.ctx)
        pointcloud_channel = PointCloudChannel(pc_topic, context=self.ctx)
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            writer = foxglove.open_mcap(path, context=self.ctx)

        os.makedirs(mesh_dir, exist_ok=True)
        copied_paths = set()

        for scene in lst:
            for o in scene:
                if "mesh_list" in o:
                    for orig_path, rel_path in o["mesh_list"]:
                        if orig_path not in copied_paths:
                            safe_copy(orig_path, os.path.join(mesh_dir, rel_path), allow_overwrite=True)
                            copied_paths.add(orig_path)

        print("start vis")

        def vis_loop(stdscr, dt=dt, lst=lst):
            stdscr.clear()
            stdscr.nodelay(1)
            cur_t = 0
            continuing = 1
            int_num = 0
            while True:
                entities, pc = to_scene_entity_pc(lst[cur_t])
                scene_channel.log(SceneUpdate(entities=[entities]))
                if pc is not None:
                    pointcloud_channel.log(pc)
                key = stdscr.getch()
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    continuing = 1 - continuing
                elif key == ord("j"):
                    continuing = False
                    cur_t = min(cur_t + 1, len(lst) - 1)
                elif key == ord("k"):
                    continuing = False
                    cur_t = max(cur_t - 1, 0)
                elif key == ord("r"):
                    continuing = False
                    cur_t = 0
                elif key == ord("b"):
                    continuing = False
                    cur_t = len(lst) - 1
                elif key >= ord("0") and key <= ord("9"):
                    int_num = int_num * 10 + (key - ord("0"))
                elif key == ord("g"):
                    continuing = False
                    cur_t = int_num
                    int_num = 0
                elif key == ord("w"):
                    dt /= 2
                elif key == ord("s"):
                    dt *= 2
                sleep(dt)
                cur_t = (cur_t + continuing) % len(lst)

        # writer.close()
        if non_blocking_t is not None:
            scene_channel.log(
                SceneUpdate(entities=[to_scene_entity(lst[non_blocking_t])])
            )
        else:
            curses.wrapper(vis_loop)

    def img(self, image: np.ndarray, topic: str = "/image"):
        img_channel = RawImageChannel(topic, context=self.ctx)
        img_channel.log(
            RawImage(
                data=image.tobytes(),
                width=image.shape[1],
                height=image.shape[0],
                encoding="rgb8",
                step=image.shape[1] * 3,
            )
        )

    def video_file(self, video_path: str, topic: str = "/video"):
        video_channel = CompressedVideoChannel(topic, context=self.ctx)
        format = get_video_codec(video_path)
        if format == "h264":
            tmp_path = os.path.join("tmp", "convert_video", gen_uuid() + ".h264")
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            os.system(
                f"ffmpeg -i {video_path} -c:v copy -bsf:v h264_mp4toannexb -f h264 {tmp_path}"
            )
            video_path = tmp_path
        with open(video_path, "rb") as f:
            video_channel.log(CompressedVideo(data=f.read(), format=format))

    def text(self, text: str, topic: str = "/text"):
        log_channel = LogChannel(topic, context=self.ctx)
        log_channel.log(Log(message=text))


if __name__ == "__main__":
    vis = Vis()
    # Vis.show([Vis.sphere(np.array([0, 0, 0]))])
    # vis.show([vis.sphere(np.array([0, 0, 0])), vis.sphere(np.array([0.1, 0, 0]), color=np.array([1, 0, 0, 1]))+vis.sphere(np.array([0.2, 0, 0]), color=np.array([0, 1, 0, 1]))])
    qpos = np.array(
        [
            0.3,
            1.2,
            0.85,
            0.0,
            0.0,
            0.0,
            -1.19,
            -0.97,
            0.06,
            -1.43,
            -1.45,
            0.37,
            -1.23,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.19,
            0.97,
            -0.06,
            1.43,
            1.45,
            -0.37,
            1.23,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    ) * 0
    urdf_path = "tmp/temp_urdf/0a9f408c-ebba-4588-8276-6face8d28b1c/robot.urdf"
    lst = []
    for i in range(10):
        qpos[0] = i * 0.
        lst.append(Vis.robot(urdf_path, qpos, mesh_type="collision", name=f"galbot"))
    vis.show(lst)
