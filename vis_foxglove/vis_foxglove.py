import os
from typing import Optional, List, Dict, Union
import numpy as np

try:
    import torch
except ImportError:
    torch = None
from time import sleep
from transforms3d.quaternions import mat2quat
import matplotlib.colors as mcolors
import foxglove
from foxglove import Context
from foxglove.channels import SceneUpdateChannel
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
)

from vis_foxglove.utils import safe_copy, rm_r, to_numpy, gen_uuid, to_number
from vis_foxglove.request import request_sync
from vis_foxglove.pin_model import PinRobotModel

if torch is not None:
    Ary = Union[np.ndarray, torch.tensor]
else:
    Ary = np.ndarray

robot_models: Dict[str, PinRobotModel] = dict()


def convert_to_rel_path(path: str) -> str:
    if path[0] != "/":
        return path
    return f"absolute_path{path}"


def to_scene_entity(scenes: List[Dict[str, list]]) -> SceneEntity:
    return SceneEntity(
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
        frame_id="<root>",
    )


class Vis:
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
    def to_color(color: Union[str, Ary] = None, opacity: float = 1.0) -> Color:
        if isinstance(color, str):
            color = mcolors.to_rgb(color)
        if color is None:
            color = np.array([1, 0, 0])
        if len(color) == 3:
            color = np.append(color, opacity)
        return Color(r=color[0], g=color[1], b=color[2], a=color[3])

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
                        size=Vector3(x=radius, y=radius, z=radius),
                        color=Vis.to_color(color, opacity),
                    )
                ]
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
        head_length = 0.02 if head_length is None else to_number(head_length)
        head_diameter = 0.02 if head_diameter is None else to_number(head_diameter)
        color = "blue" if color is None else color
        rot = np.zeros((3, 3))
        direction = direction / np.linalg.norm(direction)
        rot[:, 0] = direction
        rot[np.abs(direction[:, 0]).argmin(), 1] = 1.0
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
        elif path.endswith(".stl"):
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
                overwrite_mesh_path = mesh_param['path'].replace(os.path.dirname(urdf), name)
                lst += Vis.mesh(
                    path=mesh_param["path"],
                    trans=rot @ mesh_trans + trans,
                    rot=rot @ mesh_rot,
                    opacity=opacity,
                    color=color,
                    overwrite_mesh_path=overwrite_mesh_path,
                )
        return lst

    @staticmethod
    def show(
        lst: List[List[Dict[str, list]]],
        path: Optional[str] = None,
        dt: float = 0.1,
        mesh_dir="tmp/vis",
    ):
        ctx = Context()
        scene_channel = SceneUpdateChannel("/scene", context=ctx)
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            writer = foxglove.open_mcap(path, context=ctx)

        rm_r(mesh_dir)
        os.makedirs(mesh_dir, exist_ok=True)
        copied_paths = set()

        for scene in lst:
            for o in scene:
                if "mesh_list" in o:
                    for orig_path, rel_path in o["mesh_list"]:
                        if orig_path not in copied_paths:
                            safe_copy(orig_path, os.path.join(mesh_dir, rel_path))
                            copied_paths.add(orig_path)

        # request_sync()

        server = foxglove.start_server(context=ctx)
        try:
            first_run = True
            while True:
                for scene in lst:
                    scene_channel.log(SceneUpdate(entities=[to_scene_entity(scene)]))
                    sleep(dt)
                if first_run and path is not None:
                    writer.close()
                first_run = False
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
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
    )
    urdf_path = "tmp/temp_urdf/fb9ab88e-e0c0-4323-8412-5a161a0e3fd9/robot.urdf"
    lst = []
    for i in range(10):
        qpos[0] = i * 0.1
        lst.append(Vis.robot(urdf_path, qpos, mesh_type="collision", name=f"galbot"))
    Vis.show(lst)
