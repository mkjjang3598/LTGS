import torch
import numpy as np
from scipy.spatial import cKDTree

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   

def sh_values(sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2] (N,3,15)
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    N, _, C = sh.shape
   
    sh_values_1 = torch.zeros((N,C,3,3), dtype=torch.float32, device=sh.device) #(N,C,3,3)
    x, y, z = dirs[..., 0:3, 0], dirs[..., 0:3, 1], dirs[..., 0:3, 2] #(1,1,3)
    sh_values_1[:,:,:,0] = - C1 * y
    sh_values_1[:,:,:,1] = C1 * z 
    sh_values_1[:,:,:,2] = - C1 * x
    sh_values_2 = torch.zeros((N,C,5,5), dtype=torch.float32, device=sh.device) #(N,C,5,5)
    x, y, z = dirs[..., 3:8, 0], dirs[..., 3:8, 1], dirs[..., 3:8, 2] #(1,1,5)
    xx, yy, zz = x * x, y * y, z * z #(1,1,5)
    xy, yz, xz = x * y, y * z, x * z #(1,1,5)
    sh_values_2[:,:,:,0] = C2[0] * xy
    sh_values_2[:,:,:,1] = C2[1] * yz
    sh_values_2[:,:,:,2] = C2[2] * (2.0 * zz - xx - yy) 
    sh_values_2[:,:,:,3] = C2[3] * xz
    sh_values_2[:,:,:,4] = C2[4] * (xx - yy)
    sh_values_3 = torch.zeros((N,C,7,7), dtype=torch.float32, device=sh.device) #(N,C,7,7)
    x, y, z = dirs[..., 8:15, 0], dirs[..., 8:15, 1], dirs[..., 8:15, 2] #(1,1,7)
    xx, yy, zz = x * x, y * y, z * z #(1,1,7)
    xy, yz, xz = x * y, y * z, x * z #(1,1,7)
    sh_values_3[:,:,:,0] = C3[0] * y * (3 * xx - yy)
    sh_values_3[:,:,:,1] = C3[1] * xy * z
    sh_values_3[:,:,:,2] = C3[2] * y * (4 * zz - xx - yy)
    sh_values_3[:,:,:,3] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
    sh_values_3[:,:,:,4] = C3[4] * x * (4 * zz - xx - yy)
    sh_values_3[:,:,:,5] = C3[5] * z * (xx - yy)
    sh_values_3[:,:,:,6] = C3[6] * x * (xx - 3 * yy)
    return sh_values_1, sh_values_2, sh_values_3

def sh_rotation(sh_rest, rotation):
    """
    sh: (N,15,3)
    rotation: (3, 3)
    """
    dirs = torch.randn(size=(15,3), device=sh_rest.device)
    dirs = (dirs / (torch.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)) # (15,3)
    
    sh_values_1, sh_values_2, sh_values_3 = sh_values(sh_rest, dirs[None, None])
    dirs_rotation = dirs @ rotation.T # (15,3)
    sh_values_1_rotation, sh_values_2_rotation, sh_values_3_rotation = sh_values(sh_rest, dirs_rotation[None, None])
    sh_transform_1 = (torch.linalg.pinv(sh_values_1) @ sh_values_1_rotation).permute(0,2,3,1) # (N,3,3,C)
    sh_transform_2 = (torch.linalg.pinv(sh_values_2) @ sh_values_2_rotation).permute(0,2,3,1) # (N,5,5,C)
    sh_transform_3 = (torch.linalg.pinv(sh_values_3) @ sh_values_3_rotation).permute(0,2,3,1) # (N,7,7,C)
    sh_rotation = torch.zeros_like(sh_rest, device=sh_rest.device)
    """ original code
    sh_rotation[:,:,0:3] = (sh[:,:,None,0:3] @ sh_transform_1)[:,:,0,:]
    sh_rotation[:,:,3:8] = (sh[:,:,None,3:8] @ sh_transform_2)[:,:,0,:]
    sh_rotation[:,:,8:15] = (sh[:,:,None,8:15] @ sh_transform_3)[:,:,0,:]
    """
    sh_rotation[:,0:3,:] = torch.einsum('bijk,bjlk->bilk', sh_rest[:,None,0:3,:], sh_transform_1).squeeze(1)
    sh_rotation[:,3:8,:] = torch.einsum('bijk,bjlk->bilk', sh_rest[:,None,3:8,:], sh_transform_2).squeeze(1)
    sh_rotation[:,8:15,:] = torch.einsum('bijk,bjlk->bilk', sh_rest[:,None,8:15,:], sh_transform_3).squeeze(1)
    return sh_rotation

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
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
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def nearest_distances_ckdtree(a: torch.Tensor, b: torch.Tensor, batch_b: int = 100_000) -> torch.Tensor:
    a_np = a.detach().cpu().numpy().astype('float32')
    tree = cKDTree(a_np)

    M = b.shape[0]
    result = torch.empty(M, device=a.device, dtype=a.dtype)

    workers_arg = {}
    try:
        # SciPy ≥1.6
        import inspect
        sig = inspect.signature(tree.query)
        if 'workers' in sig.parameters:
            workers_arg = {'workers': -1}
    except Exception:
        pass

    for i in range(0, M, batch_b):
        b_chunk = b[i:i+batch_b].detach().cpu().numpy().astype('float32')
        dists, _ = tree.query(b_chunk, k=1, **workers_arg)
        result[i:i+batch_b] = torch.from_numpy(dists).to(a.device)

    return result

# function from 3DGS-CD
def exp_map_SO3xR3(tangent_vector: torch.Tensor) -> torch.Tensor:
    """Compute the exponential map of the direct product group `SO(3) x R^3`.
    This can be used for learning pose deltas on SE(3), and is generally faster than `exp_map_SE3`.
    Args:
        tangent_vector: Tangent vector; length-3 translations, followed by an `so(3)` tangent vector (..., 6).
    Returns:
        [R|t] transformation matrices (..., 3, 4).
    """
    # code for SO3 map grabbed from pytorch3d and stripped down to bare-bones
    log_rot = tangent_vector[:, 3:]
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, 1e-4).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = torch.zeros((log_rot.shape[0], 3, 3), dtype=log_rot.dtype, device=log_rot.device)
    skews[:, 0, 1] = -log_rot[:, 2]
    skews[:, 0, 2] = log_rot[:, 1]
    skews[:, 1, 0] = log_rot[:, 2]
    skews[:, 1, 2] = -log_rot[:, 0]
    skews[:, 2, 0] = -log_rot[:, 1]
    skews[:, 2, 1] = log_rot[:, 0]
    skews_square = torch.bmm(skews, skews)

    ret = torch.zeros(tangent_vector.shape[0], 3, 4, dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    # Compute the translation
    ret[:, :3, 3] = tangent_vector[:, :3]
    return ret