import os
from typing import Any, Optional, List, Dict, Union
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
from foxglove.schemas import SpherePrimitive, Pose, Vector3, Quaternion, Color, SceneUpdate, SceneEntity, ModelPrimitive

from vis_foxglove.utils import safe_copy, rm_r, to_numpy, gen_uuid
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
        arrows=[arrow for scene in scenes for arrow in scene.get('arrows', [])],
        cubes=[cube for scene in scenes for cube in scene.get('cubes', [])],
        spheres=[sphere for scene in scenes for sphere in scene.get('spheres', [])],
        cylinders=[cylinder for scene in scenes for cylinder in scene.get('cylinders', [])],
        lines=[line for scene in scenes for line in scene.get('lines', [])],
        triangles=[triangle for scene in scenes for triangle in scene.get('triangles', [])],
        texts=[text for scene in scenes for text in scene.get('texts', [])],
        models=[model for scene in scenes for model in scene.get('models', [])],
    )
        
class Vis:
    @staticmethod
    def to_pose(trans: Ary, rot: Optional[Ary] = None) -> Pose:
        if rot is None:
            rot = np.eye(3)
        wxyz = mat2quat(rot)
        return Pose(
            position=Vector3(x=trans[0], y=trans[1], z=trans[2]),
            orientation=Quaternion(x=wxyz[1], y=wxyz[2], z=wxyz[3], w=wxyz[0])
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
    
    def sphere(self, 
               trans: Ary = None, 
               radius: float = None, 
               color: Union[str, Ary] = None, 
               opacity: float = None
        ) -> List[Dict[str, list]]:
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        radius = 0.1 if radius is None else to_number(radius)
        color = 'blue' if color is None else color
        opacity = 1.0 if opacity is None else opacity
        return [dict(
                spheres=[
                    SpherePrimitive(
                        pose=self.to_pose(trans),
                        size=Vector3(x=radius, y=radius, z=radius),
                        color=self.to_color(color, opacity)
                    )
                ]
        )]
    
    @staticmethod
    def mesh(path: str, 
             trans: Ary = None, 
             rot: Ary = None, 
             scale: Union[float, Ary] = None, 
             color: Optional[Union[str, Ary]] = None,
             opacity: float = None
        ) -> List[Dict[str, list]]:
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        rot = np.eye(3) if rot is None else to_numpy(rot)
        if isinstance(scale, float):
            scale = np.array([scale, scale, scale])
        scale = np.ones(3) if scale is None else to_numpy(scale)
        color = 'pink' if color is None else color
        opacity = 1.0 if opacity is None else opacity
        
        if path.startswith("http://"):
            url = path
            mesh_set = set()
        else:
            new_path = convert_to_rel_path(path)
            url = f'http://localhost:19685/{new_path}'
            mesh_set = set([path])
        
        return [dict(
                models=[
                    ModelPrimitive(
                        pose=Vis.to_pose(trans),
                        scale=Vector3(x=scale[0], y=scale[1], z=scale[2]),
                        url=url,
                        color=Vis.to_color(color, opacity)
                    )
                ],
                mesh_set=mesh_set,
        )]
    
    @staticmethod
    def robot(urdf: str, 
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
                    name=f"{name}_sphere_id{mesh_id}",
                )
            elif mesh_type == "mesh":
                lst += Vis.mesh(
                    path=mesh_param["path"],
                    trans=rot @ mesh_trans + trans,
                    rot=rot @ mesh_rot,
                    opacity=opacity,
                    color=color,
                    name=f"{name}_mesh_id{mesh_id}",
                )
        return lst
        
    @staticmethod
    def show(lst: List[List[Dict[str, list]]], path: Optional[str] = None, dt: float = 0.1, mesh_dir='tmp/vis'):
        ctx = Context()
        scene_channel = SceneUpdateChannel("/scene", context=ctx)
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            writer = foxglove.open_mcap(path, context=ctx)

        full_mesh_set, full_urdf_dict = set(), dict()
        for scene in lst:
            for o in scene:
                if "mesh_set" in o:
                    full_mesh_set.update(o["mesh_set"])
                if "urdf" in o:
                    urdf_name, urdf_path = o['urdf']['name'], o['urdf']['path']
                    if urdf_name in full_urdf_dict:
                        assert full_urdf_dict[urdf_name] == urdf_path
                    else:
                        full_urdf_dict[urdf_name] = urdf_path

        rm_r(mesh_dir)
        os.makedirs(mesh_dir, exist_ok=True)
        for orig_path in full_mesh_set:
            rel_path = convert_to_rel_path(orig_path)
            safe_copy(orig_path, os.path.join(mesh_dir, rel_path))
        
        # for urdf_name, urdf_path in full_urdf_dict.items():
        #     urdf_dir = os.path.dirname(urdf_path)
        #     rel_save_path = os.path.join(urdf_name, os.path.basename(urdf_path))
        #     safe_copy(
        #         urdf_dir, os.path.join(mesh_dir, os.path.dirname(rel_save_path))
        #     )
        
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
    Vis.show([Vis.sphere(np.array([0, 0, 0]))])
    # vis.show([vis.sphere(np.array([0, 0, 0])), vis.sphere(np.array([0.1, 0, 0]), color=np.array([1, 0, 0, 1]))+vis.sphere(np.array([0.2, 0, 0]), color=np.array([0, 1, 0, 1]))])
    