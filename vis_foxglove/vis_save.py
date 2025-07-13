import os
from typing import Optional, Union, Dict, List
import trimesh as tm
import numpy as np
import torch
import json
from transforms3d.quaternions import mat2quat

from .utils import (
    to_numpy,
    to_torch,
    to_number,
    rm_r,
    safe_copy,
    serialize_item,
    gen_uuid,
)
from .pin_model import PinRobotModel


def convert_to_rel_path(path: str) -> str:
    if path[0] != "/":
        return path
    return f"absolute_path{path}"


class Vis:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def save(lst: Union[List[Dict], List[List[Dict]]], dir: str = "tmp/vis_0"):
        if isinstance(lst[0], dict):
            lst = [lst]
        # empty directory
        rm_r(dir)
        os.makedirs(dir, exist_ok=True)
        for lt in lst:
            for o in lt:
                if not o.get("path") is None:
                    rel_save_path = convert_to_rel_path(o["path"])
                    safe_copy(o["path"], os.path.join(dir, rel_save_path))
                    o["path"] = rel_save_path
                if not o.get("urdf") is None:
                    urdf_dir = os.path.dirname(o["urdf"])
                    rel_save_path = os.path.join(o["name"], os.path.basename(o["urdf"]))
                    safe_copy(
                        urdf_dir, os.path.join(dir, os.path.dirname(rel_save_path))
                    )
                    o["urdf"] = rel_save_path
        with open(os.path.join(dir, "scene.json"), "w") as f:
            json.dump(serialize_item(lst), f, indent=4)
        print(f"Saved to {dir}")

    @staticmethod
    def sphere(
        trans: Optional[Union[np.ndarray, torch.tensor]] = None,
        radius: float = None,
        opacity: float = None,
        color: str = None,
        name: str = None,
    ) -> list:
        color = "blue" if color is None else color
        opacity = 1.0 if opacity is None else opacity
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        radius = 0.1 if radius is None else to_number(radius)
        name = gen_uuid() if name is None else name

        return [
            dict(
                type="sphere",
                color=color,
                opacity=opacity,
                radius=radius,
                trans=trans,
                name=name,
            )
        ]

    @staticmethod
    def ellipsoid(
        trans: Optional[Union[np.ndarray, torch.tensor]] = None,
        rot: Optional[Union[np.ndarray, torch.tensor]] = None,
        axis: Optional[Union[np.ndarray, torch.tensor]] = None,
        subdivisions: int = 4,
        opacity: float = None,
        color: str = None,
        name: str = None,
    ) -> list:
        color = "green" if color is None else color
        opacity = 1.0 if opacity is None else opacity
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        rot = np.eye(3) if rot is None else to_numpy(rot)
        axis = np.array([1, 1, 1]) if axis is None else to_numpy(axis)
        name = gen_uuid() if name is None else name

        sphere = tm.creation.icosphere(subdivisions=subdivisions, radius=1.0)

        # Apply scaling along each axis to transform the sphere into an ellipsoid
        scaling_matrix = np.diag([axis[0], axis[1], axis[2], 1])
        sphere.apply_transform(scaling_matrix)
        v, f = sphere.vertices, sphere.faces
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = trans

        return [
            dict(
                type="mesh",
                path=None,
                urdf=None,
                scale=1.0,
                pos=trans,
                quat=mat2quat(rot),
                opacity=opacity,
                color=color,
                vertices=v,
                faces=f,
                pose=pose,
                name=name,
            )
        ]

    @staticmethod
    def box(
        scale: Union[np.ndarray, torch.tensor],  # (3, )
        trans: Optional[Union[np.ndarray, torch.tensor]] = None,  # (3, )
        rot: Optional[Union[np.ndarray, torch.tensor]] = None,  # (3, 3)
        opacity: Optional[float] = None,
        color: Optional[str] = None,
        name: str = None,
    ) -> list:

        color = "violet" if color is None else color
        opacity = 1.0 if opacity is None else opacity
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        rot = np.eye(3) if rot is None else to_numpy(rot)
        name = gen_uuid() if name is None else name

        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = trans

        return [
            dict(
                type="box",
                color=color,
                opacity=opacity,
                size=scale,
                pos=trans,
                quat=mat2quat(rot),
                pose=pose,
                name=name,
            )
        ]

    @staticmethod
    def pose(
        trans: Union[np.ndarray, torch.tensor],  # (3, )
        rot: Union[np.ndarray, torch.tensor],  # (3, 3)
        width: int = 0.1,
        length: float = 0.01,
        name: str = None,
    ) -> list:
        name = gen_uuid() if name is None else name
        return [
            dict(
                type="pose", trans=trans, rot=rot, width=width, length=length, name=name
            )
        ]

    @staticmethod
    def plane(
        plane_vec: Union[np.ndarray, torch.tensor],  # (4, )
        opacity: Optional[float] = None,
        color: Optional[str] = None,
        name: str = None,
    ) -> list:
        plane_vec = to_torch(plane_vec)
        color = "blue" if color is None else color
        opacity = 1.0 if opacity is None else opacity
        dir = plane_vec[:3]
        name = gen_uuid() if name is None else name
        assert (torch.linalg.norm(dir) - 1).abs() < 1e-4
        return [
            dict(type="plane", color=color, opacity=opacity, plane=plane_vec, name=name)
        ]

    @staticmethod
    def gripper(
        trans: Optional[Union[np.ndarray, torch.tensor]] = None,  # (3)
        rot: Optional[Union[np.ndarray, torch.tensor]] = None,  # (3, 3)
        width: Optional[float] = 0.0,
        depth: Optional[float] = 0.0,
        height: Optional[float] = 0.002,
        finger_width: Optional[float] = 0.002,
        tail_length: Optional[float] = 0.04,
        depth_base: Optional[float] = 0.06,
        opacity: Optional[float] = None,
        color: Optional[str] = None,
    ):
        """
        4 boxes:
                   2|------ 1
            --------|  . O
                3   |------ 0

                                    y
                                    |
                                    O--x
                                   /
                                  z
        """
        trans = np.zeros((3,)) if trans is None else to_numpy(trans).reshape(3)
        rot = np.eye(3) if rot is None else to_numpy(rot).reshape(3, 3)
        color = "blue" if color is None else color
        opacity = 1.0 if opacity is None else opacity

        centers = np.array(
            [
                [
                    (depth - finger_width - depth_base) / 2,
                    (width + finger_width) / 2,
                    0,
                ],
                [
                    (depth - finger_width - depth_base) / 2,
                    -(width + finger_width) / 2,
                    0,
                ],
                [-depth_base - finger_width / 2, 0, 0],
                [-depth_base - finger_width - tail_length / 2, 0, 0],
            ]
        )
        scales = np.array(
            [
                [finger_width + depth_base + depth, finger_width, height],
                [finger_width + depth_base + depth, finger_width, height],
                [finger_width, width, height],
                [tail_length, finger_width, height],
            ]
        )
        centers = np.einsum("ij,kj->ki", rot, centers) + trans
        box_plotly_list = []
        for i in range(4):
            box_plotly_list += Vis.box(scales[i], centers[i], rot, opacity, color)
        return box_plotly_list

    @staticmethod
    def robot(
        urdf: str,
        qpos: Union[Union[np.ndarray, torch.tensor]],  # (n)
        trans: Optional[Union[np.ndarray, torch.tensor]] = None,  # (3)
        rot: Optional[Union[np.ndarray, torch.tensor]] = None,  # (3, 3)
        opacity: Optional[float] = None,
        color: Optional[str] = None,
        mesh_type: str = "visual",
        name: str = None,
        unpack: bool = False,
    ) -> list:
        trans = np.zeros((3,)) if trans is None else to_numpy(trans).reshape(3)
        rot = np.eye(3) if rot is None else to_numpy(rot).reshape(3, 3)
        qpos = to_numpy(qpos).reshape(-1)
        color = "violet" if color is None else color
        opacity = 1.0 if opacity is None else opacity
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = trans
        name = gen_uuid() if name is None else name

        if unpack:
            pin_model = PinRobotModel(urdf)
            poses = pin_model.fk_mesh(qpos, mode=mesh_type)
            lst = []
            for mesh_id, ((mesh_type, mesh_param), (mesh_trans, mesh_rot)) in enumerate(
                zip(pin_model.meshes[mesh_type], poses)
            ):
                if mesh_type == "sphere":
                    lst += Vis.sphere(
                        trans=rot @ mesh_trans + trans,
                        radius=mesh_param["radius"],
                        opacity=opacity,
                        color=color,
                        name=f"{name}_sphere_id{mesh_id}",
                    )
                elif mesh_type == "mesh":
                    vertices, faces = (
                        mesh_param["mesh"].vertices,
                        mesh_param["mesh"].faces,
                    )
                    lst += Vis.mesh(
                        vertices=vertices,
                        faces=faces,
                        trans=rot @ mesh_trans + trans,
                        rot=rot @ mesh_rot,
                        opacity=opacity,
                        color=color,
                        name=f"{name}_mesh_id{mesh_id}",
                    )
            return lst
        else:
            return [
                dict(
                    type="robot",
                    urdf=urdf,
                    color=color,
                    opacity=opacity,
                    qpos=qpos,
                    pos=trans,
                    quat=mat2quat(rot),
                    pose=pose,
                    mesh_type=mesh_type,
                    name=name,
                )
            ]

    @staticmethod
    def pc(
        pc: Union[np.ndarray, torch.tensor],  # (n, 3)
        value: Optional[Union[np.ndarray, torch.tensor]] = None,  # (n, )
        size: int = 1,
        color: Union[str, Union[np.ndarray, torch.tensor]] = "red",  # (n, 3)
        color_map: str = "Viridis",
        name: str = None,
    ) -> list:
        pc = to_numpy(pc)
        if not value is None:
            value = to_numpy(value)
        if not isinstance(color, str):
            color = to_numpy(color)
        name = gen_uuid() if name is None else name

        return [
            dict(
                type="pc",
                pc=pc,
                value=value,
                size=size,
                color=color,
                color_map=color_map,
                name=name,
            )
        ]

    @staticmethod
    def line(
        p1: Union[np.ndarray, torch.tensor],  # (3)
        p2: Union[np.ndarray, torch.tensor],  # (3)
        width: int = None,
        color: str = None,
        name: str = None,
    ) -> list:
        color = "green" if color is None else color
        width = 1 if width is None else width
        name = gen_uuid() if name is None else name

        p1, p2 = to_numpy(p1), to_numpy(p2)
        return [dict(type="line", p1=p1, p2=p2, width=width, color=color, name=name)]

    @staticmethod
    def mesh(
        path: str = None,
        scale: float = 1.0,
        trans: Optional[Union[np.ndarray, torch.tensor]] = None,  # (3, )
        rot: Optional[Union[np.ndarray, torch.tensor]] = None,  # (3, 3)
        opacity: float = 1.0,
        color: str = "lightgreen",
        vertices: Optional[Union[np.ndarray, torch.tensor]] = None,  # (n, 3)
        faces: Optional[Union[np.ndarray, torch.tensor]] = None,  # (m, 3)
        name: str = None,
    ) -> list:
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        rot = np.eye(3) if rot is None else to_numpy(rot)
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = trans
        name = gen_uuid() if name is None else name
        if path is None:
            vertices = to_numpy(vertices)
            faces = to_numpy(faces)

        return [
            dict(
                type="mesh",
                path=path,
                urdf=None,
                scale=scale,
                pos=trans,
                quat=mat2quat(rot),
                opacity=opacity,
                color=color,
                vertices=vertices,
                faces=faces,
                pose=pose,
                name=name,
            )
        ]

    @staticmethod
    def self_test():
        from pytorch3d import transforms as pttf

        # each element in all_vis represents one timestep of the scene
        all_vis = []
        # one timestep can have multiple objects, we can add the lists of objects together
        all_vis.append(
            Vis.sphere(trans=np.random.randn(3) / 10, opacity=0.9)
            + Vis.box(
                trans=np.random.randn(3) / 10,
                rot=pttf.random_rotation(),
                scale=np.random.rand(3) / 10 + 0.001,
                opacity=0.9,
            )
        )
        all_vis.append(
            Vis.box(
                trans=np.random.randn(3) / 10,
                rot=pttf.random_rotation(),
                scale=np.random.rand(3) / 10 + 0.001,
                opacity=0.9,
            )
        )
        all_vis.append(
            Vis.pose(trans=np.random.randn(3) / 10, rot=pttf.random_rotation())
        )
        all_vis.append(
            Vis.plane(
                torch.cat(
                    [
                        pttf.random_rotation()[0],
                        torch.randn(
                            1,
                        )
                        / 10,
                    ]
                ),
                opacity=0.9,
            )
        )
        all_vis.append(Vis.pc(np.random.randn(100, 3) / 10))
        all_vis.append(Vis.pc(np.random.randn(100, 3) / 10, value=np.random.randn(100)))
        all_vis.append(
            Vis.pc(np.random.randn(100, 3) / 10, color=np.random.rand(100, 3))
        )
        all_vis.append(
            Vis.pc(np.random.randn(100, 3) / 10, color=np.random.rand(100, 4))
        )
        all_vis.append(Vis.line(np.random.randn(3) / 10, np.random.randn(3) / 10))
        Vis.save(all_vis, "tmp/vis")


if __name__ == "__main__":
    Vis.self_test()
