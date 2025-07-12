import sys
import os
import torch
import numpy as np
import random
from typing import Union
from datetime import datetime
import uuid
from multiprocessing import Pool
import shutil
import filecmp
import trimesh as tm


def to_list(x: Union[torch.Tensor, np.ndarray, list], spec="cpu") -> list:
    if isinstance(x, torch.Tensor):
        return x.tolist()
    elif isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, list):
        return x
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def to_torch(x: Union[torch.Tensor, np.ndarray, list], spec="cpu") -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(spec)
    elif isinstance(x, torch.Tensor):
        return x.to(spec)
    elif isinstance(x, list):
        return torch.tensor(x).to(spec)
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def to_numpy(x: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return np.array(x)
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def to_number(x: Union[torch.Tensor, np.ndarray, float, int]) -> float:
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return x.item()
    return float(x)


def torch_dict_to_device(dic, device):
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in dic.items()
    }


def transform_pc(
    pose: Union[torch.Tensor, np.ndarray],
    pc: Union[torch.Tensor, np.ndarray],
    inv: bool = False,
):
    # pose: ([B]*, 4, 4), pc: ([B]*, N, 3) -> result ([B]*, N, 3)
    orig_shape = pose.shape[:-2]
    pose, pc = pose.reshape(-1, 4, 4), pc.reshape(-1, pc.shape[-2], 3)
    einsum = np.einsum if isinstance(pose, np.ndarray) else torch.einsum
    if inv:
        result = einsum("bji,bni->bnj", pose[:, :3, :3], pc - pose[:, :3, 3][:, None])
    else:
        result = einsum("bij,bnj->bni", pose[:, :3, :3], pc) + pose[:, :3, 3][:, None]
    return result.reshape(*orig_shape, -1, 3)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def to_voxel_center(pc: torch.Tensor, voxel_size: float):
    """calculate the center of voxel corresponding to each point

    Args:
        pc (torch.Tensor): (..., 3)
    returns:
        voxel_center (torch.Tensor): (..., 3)
    """
    return (
        torch.div(pc, voxel_size, rounding_mode="floor") * voxel_size + voxel_size / 2
    )


def proper_svd(rot: torch.Tensor):
    """
    compute proper svd of rotation matrix
    rot: (B, 3, 3)
    return rotation matrix (B, 3, 3) with det = 1
    """
    u, s, v = torch.svd(rot.double())
    with torch.no_grad():
        sign = torch.sign(torch.det(torch.einsum("bij,bkj->bik", u, v)))
        diag = torch.stack(
            [torch.ones_like(s[:, 0]), torch.ones_like(s[:, 1]), sign], dim=-1
        )
        diag = torch.diag_embed(diag)
    return torch.einsum("bij,bjk,blk->bil", u, diag, v).to(rot.dtype)


def silent_call(func, *args, **kwargs):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(e)
        raise e
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    return result


def depth_img_to_xyz(depth_img: np.ndarray, intrinsics: np.ndarray):
    """
    depth_img: (H, W)
    intrinsics: (3, 3)
    return: (H, W, 3)
    """
    H, W = depth_img.shape
    x = np.arange(W)
    y = np.arange(H)
    x, y = np.meshgrid(x, y)
    x = (x - intrinsics[0, 2]) * depth_img / intrinsics[0, 0]
    y = (y - intrinsics[1, 2]) * depth_img / intrinsics[1, 1]
    return np.stack([x, y, depth_img], -1)


def get_time_str():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def sample_sphere(center: np.ndarray, radius: float, n: int = 1):
    theta = np.random.rand(n) * np.pi * 2
    r = radius * np.sqrt(np.random.rand(n))
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1) * r[:, None] + center


def gen_uuid():
    return str(uuid.uuid4())


def inf_generator():
    while True:
        yield


def pool_process(num, func, args):
    pool = Pool(processes=num)
    results = pool.map(func, args)
    pool.close()
    pool.join()
    # results = pool.starmap(func, args)
    return results


# def matrix_to_axis_angle(rot: torch.Tensor):
#     aa = pttf.matrix_to_axis_angle(rot.reshape(-1, 3, 3))
#     aa_norm = aa.norm(dim=-1, keepdim=True)
#     aa = torch.where(aa_norm < np.pi, aa, aa / aa_norm * (aa_norm - 2 * np.pi))
#     return aa.reshape(*rot.shape[:-2], 3)


# def cdist_test(pc1, pc2, thresh, filter=True):
#     if filter:
#         pc1_min, pc1_max = pc1.min(dim=0).values, pc1.max(dim=0).values
#         for i in range(3):
#             pc2 = pc2[
#                 (pc2[:, i] > pc1_min[i] - 1.2 * thresh)
#                 & (pc2[:, i] < pc1_max[i] + 1.2 * thresh)
#             ]
#             if len(pc2) == 0:
#                 return False
#         pc2_min, pc2_max = pc2.min(dim=0).values, pc2.max(dim=0).values
#         for i in range(3):
#             pc1 = pc1[
#                 (pc1[:, i] > pc2_min[i] - 1.2 * thresh)
#                 & (pc1[:, i] < pc2_max[i] + 1.2 * thresh)
#             ]
#             if len(pc1) == 0:
#                 return False
#     # cdist = torch.cdist(pc1, pc2)
#     # return cdist.min() < thresh
#     if len(pc2) > len(pc1):
#         pc1, pc2 = pc2, pc1
#     tree = cKDTree(pc2)
#     distances, _ = tree.query(
#         pc1, k=1
#     )  # Find the nearest point in pc2 for each point in pc1
#     return np.min(distances) < thresh


def rm_r(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def safe_copy(src, dst, allow_overwrite=False):
    # copy src to dst, allow file & dir
    if os.path.exists(dst):
        if not allow_overwrite:
            # judge whether the same file
            # Check if src and dst are the same file or directory
            if os.path.isfile(src) and os.path.isfile(dst):
                if filecmp.cmp(src, dst, shallow=False):  # Perform a deep comparison
                    return  # They are the same, do nothing
            elif os.path.isdir(src) and os.path.isdir(dst):
                # Compare directories for equality
                if filecmp.dircmp(
                    src, dst
                ).common_files:  # Add a deeper check if needed
                    return  # They are the same, do nothing
            raise FileExistsError(f"{dst} already exists")
        else:
            rm_r(dst)
    parent_dir = os.path.dirname(dst)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    copy = shutil.copy if os.path.isfile(src) else shutil.copytree
    copy(src, dst)


def serialize_item(item):
    if isinstance(item, np.ndarray):
        return item.tolist()  # Convert numpy array to list
    elif isinstance(item, torch.Tensor):
        return item.detach().cpu().tolist()  # Convert PyTorch tensor to list
    elif isinstance(item, list):  # Handle nested lists
        return [serialize_item(i) for i in item]
    elif isinstance(item, dict):  # Handle dictionaries
        return {k: serialize_item(v) for k, v in item.items()}
    return item  # Return other types as-is


def get_vertices_faces(file_name=None, mesh=None):
    if mesh is None:
        mesh = tm.load(file_name)
    if not hasattr(mesh, "vertices"):
        mesh = mesh.to_mesh()
    return mesh.vertices, mesh.faces


def save_obj_file(v, f, file_name=None):
    if file_name is None:
        file_name = f"tmp/mesh/{gen_uuid()}.obj"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    tm.Trimesh(vertices=v, faces=f).export(file_name)
    return file_name


def set_minus(a, b):
    return [x for x in a if x not in b]


def angle_to_rot33(angle: float) -> np.ndarray:  # (3, 3)
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def trans_rot_to_4x4mat(trans: np.ndarray, rot: np.ndarray) -> np.ndarray:  # (4, 4)
    res = np.eye(4)
    res[:3, :3] = rot
    res[:3, 3] = trans
    return res


# from pytorch3d
def quaternion_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = (
        quaternions[..., 0],
        quaternions[..., 1],
        quaternions[..., 2],
        quaternions[..., 3],
    )
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(axis=-1)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def load_dataclass(cls, dict: dict):
    return cls(**dict)


def zfill(x: Union[int, str], length: int) -> str:
    return str(x).zfill(length)


def rotation_angle(rot1: np.ndarray, rot2: np.ndarray) -> float:
    delta_rot = np.dot(rot1.T, rot2)
    return np.arccos(np.clip((np.trace(delta_rot) - 1) / 2, -1, 1))
