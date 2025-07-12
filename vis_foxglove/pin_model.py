from os.path import dirname
import warnings
import numpy as np
import trimesh as tm
import pinocchio
from typing import Dict, Tuple, List, Optional

from .utils import to_torch


class PinRobotModel:
    urdf_path: str
    model: pinocchio.Model
    collision_model: pinocchio.GeometryModel
    visual_model: pinocchio.GeometryModel
    data: pinocchio.Data
    collision_data: pinocchio.GeometryData
    visual_data: pinocchio.GeometryData
    # list[(N, 3) np.ndarray]
    fps_list: Optional[List[np.ndarray]]
    joint_names: List[str]
    joint_lower_limits: np.ndarray
    joint_upper_limits: np.ndarray
    # dict[collision/visual, list[mesh_id]]
    mesh_id: Dict[str, List[int]]
    # dict[collision/visual, list[(type, dict)]]
    meshes: Dict[str, List[Tuple[str, Dict]]]
    camera_link: str
    camera_trans: np.ndarray
    camera_rot: np.ndarray

    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.model, self.collision_model, self.visual_model = (
            pinocchio.buildModelsFromUrdf(urdf_path, dirname(urdf_path))
        )

        # Create data required by the algorithms
        self.data, self.collision_data, self.visual_data = pinocchio.createDatas(
            self.model, self.collision_model, self.visual_model
        )

        self.fps_list = None
        self.setup_mesh()
        self.joint_names = [n for n in self.model.names[1:]]
        self.joint_lower_limits = self.model.lowerPositionLimit
        self.joint_upper_limits = self.model.upperPositionLimit

    def setup_mesh(self, vis_links=None, fps_path=None):
        self.mesh_id = dict(collision=[], visual=[])
        self.meshes = dict(collision=[], visual=[])
        self.fps_list = []
        if not fps_path is None:
            fps = np.load(fps_path)
        for mode, model in [
            ("collision", self.collision_model),
            ("visual", self.visual_model),
        ]:
            for mesh_id, obj in enumerate(model.geometryObjects):
                if not vis_links is None:
                    if self.get_frame_name(obj.parentFrame) not in vis_links:
                        continue
                if obj.meshPath == "SPHERE":
                    self.meshes[mode].append(
                        (
                            "sphere",
                            dict(radius=obj.geometry.radius, frame_id=obj.parentFrame),
                        )
                    )
                    assert not (
                        mode == "visual" and fps_path is None
                    ), "fps_path must be provided for sphere"
                else:
                    mesh = tm.load_mesh(obj.meshPath)
                    if not hasattr(mesh, "vertices"):
                        mesh = mesh.to_mesh()
                    self.meshes[mode].append(
                        (
                            "mesh",
                            dict(
                                path=obj.meshPath, mesh=mesh, frame_id=obj.parentFrame
                            ),
                        )
                    )
                    if mode == "visual" and not fps_path is None:
                        assert obj.meshPath in fps, f"fps for {obj.meshPath} not found"
                        self.fps_list.append(
                            fps[obj.meshPath].copy().astype(np.float32)
                        )
                self.mesh_id[mode].append(mesh_id)

    def get_frame_name(self, frame_id: int) -> str:
        return self.model.frames[frame_id].name

    def get_frame_id(self, frame_name: str) -> int:
        return self.model.getFrameId(frame_name)

    def set_fk(self, q: np.ndarray) -> None:  # q: joint angles (J,)
        pinocchio.forwardKinematics(self.model, self.data, q)

    def fk_all_link(
        self, q: np.ndarray, set_fk: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        # q: joint angles (J,)
        # return link trans (str -> (3, )) and link rot (str -> (3, 3))
        if set_fk:
            self.set_fk(q)
        pinocchio.updateFramePlacements(self.model, self.data)
        link_trans = {
            self.get_frame_name(i): self.data.oMf[i].translation.copy()
            for i in range(len(self.model.frames))
        }
        link_rot = {
            self.get_frame_name(i): self.data.oMf[i].rotation.copy()
            for i in range(len(self.model.frames))
        }
        return link_trans, link_rot

    def fk_link(
        self, q: np.ndarray, link: str, set_fk: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        # q: joint angles (J,)
        # return link trans (3, ) and link rot (3, 3)
        if set_fk:
            self.set_fk(q)
        pinocchio.updateFramePlacements(self.model, self.data)
        frame = self.get_frame_id(link)
        return (
            self.data.oMf[frame].translation.copy(),
            self.data.oMf[frame].rotation.copy(),
        )

    def fk_all_jacobian(
        self, q: np.ndarray, set_fk: bool = True
    ) -> Dict[str, np.ndarray]:
        # q: joint angles (J,)
        # return link jacobian (str -> (6, J))
        if set_fk:
            self.set_fk(q)
        pinocchio.updateFramePlacements(self.model, self.data)
        return {
            self.get_frame_name(i): pinocchio.computeFrameJacobian(
                self.model, self.data, q, i
            ).copy()
            for i in range(len(self.model.frames))
        }

    def fk_jacobian(
        self, q: np.ndarray, link: str, set_fk: bool = True, mode_str="local"
    ) -> np.ndarray:
        # q: joint angles (J,)
        # return link jacobian (6, J)
        if set_fk:
            self.set_fk(q)
        i = self.get_frame_id(link)
        if mode_str == "local":
            mode = pinocchio.LOCAL
        elif mode_str == "world":
            mode = pinocchio.WORLD
        else:
            mode = pinocchio.LOCAL_WORLD_ALIGNED
        return pinocchio.computeFrameJacobian(self.model, self.data, q, i, mode).copy()

    def fk_mesh(
        self, q: np.ndarray, mode: str, set_fk: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        # q: joint angles (J,)
        # mode: "collision" or "visual"
        if set_fk:
            self.set_fk(q)
        if mode == "visual":
            pinocchio.updateGeometryPlacements(
                self.model, self.data, self.visual_model, self.visual_data
            )
            d = self.visual_data
        elif mode == "collision":
            pinocchio.updateGeometryPlacements(
                self.model, self.data, self.collision_model, self.collision_data
            )
            d = self.collision_data
        else:
            raise ValueError(f"mode must be visual/collision. current mode: {mode}")
        return [
            (d.oMg[i].translation.copy(), d.oMg[i].rotation.copy())
            for i in self.mesh_id[mode]
        ]

    def fk_fps(self, q: np.ndarray, set_fk: bool = True) -> np.ndarray:
        # q: joint angles (J,)
        # return fps (N, 3)
        if set_fk:
            self.set_fk(q)
        fk_meshs = self.fk_mesh(q, "visual", set_fk=False)
        fps = []
        for i, (trans, rot) in enumerate(fk_meshs):
            fps.append(self.fps_list[i] @ rot.T + trans)
        return np.concatenate(fps, axis=0)

    def set_camera(self, link: str, trans: np.ndarray, rot: np.ndarray) -> None:
        # link: link name
        # trans: translation (3, )
        # rot: rotation (3, 3)
        self.camera_link = link
        self.camera_trans = trans
        self.camera_rot = rot

    def fk_camera(self, q: np.ndarray, set_fk: bool = True) -> np.ndarray:
        # q: joint angles (J,)
        # return camera extrinsics (4, 4)
        lt, lr = self.fk_link(q, self.camera_link, set_fk=set_fk)
        result = np.eye(4)
        result[:3, 3] = lt + lr @ self.camera_trans
        result[:3, :3] = lr @ self.camera_rot
        # TODO: temp
        result[:3, :3] = np.array(
            [
                [0, -np.sqrt(2) / 2, np.sqrt(2) / 2],
                [-1, 0, 0],
                [0, -np.sqrt(2) / 2, -np.sqrt(2) / 2],
            ]
        )

        return result.astype(np.float32)

    def forward_kinematics(self, q: np.ndarray, mode: str, link: str = None):
        warnings.warn("This function is deprecated. Use others (if exists) instead.")
        pinocchio.forwardKinematics(self.model, self.data, q)
        if mode == "link":
            pinocchio.updateFramePlacements(self.model, self.data)
            if link is not None:
                frame = self.model.getFrameId(link)
                return (
                    self.data.oMf[frame].translation,
                    self.data.oMf[frame].rotation,
                )
            else:
                link_trans = {
                    self.model.frames[i]
                    .name: to_torch(self.data.oMf[i].translation)
                    .float()
                    for i in range(len(self.model.frames))
                }
                link_rot = {
                    self.model.frames[i]
                    .name: to_torch(self.data.oMf[i].rotation)
                    .float()
                    for i in range(len(self.model.frames))
                }
                return link_trans, link_rot
        elif mode == "visual":
            pinocchio.updateGeometryPlacements(
                self.model, self.data, self.visual_model, self.visual_data
            )
            d = self.visual_data
        elif mode == "collision":
            pinocchio.updateGeometryPlacements(
                self.model, self.data, self.collision_model, self.collision_data
            )
            d = self.collision_data
        elif mode == "jacobian":
            if link is not None:
                i = self.model.getFrameId(link)
                return pinocchio.computeFrameJacobian(self.model, self.data, q, i)
            return {
                self.model.frames[i].name: pinocchio.computeFrameJacobian(
                    self.model, self.data, q, i
                )
                for i in range(len(self.model.frames))
            }
        else:
            raise ValueError(f"mode must be visual/collision. current mode: {mode}")
        return [(oMg.translation, oMg.rotation) for oMg in d.oMg]

    def generate_random_qpos(self):
        return pinocchio.randomConfiguration(self.model)
