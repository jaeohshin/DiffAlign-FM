import torch
from torch import nn
from torch_scatter import scatter_add
import math

# --- Assume previous Quaternion Utilities (q_mult, q_inv, etc.) are defined ---
# ... (q_mult, q_inv, q_rotate, normalize_q, rot_to_quat, quat_to_rot, axis_angle_to_quat 코드는 그대로 둡니다) ...
@torch.compile
def q_mult(q1, q2):
    """Multiply two quaternions batch-wise."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)

@torch.compile
def q_inv(q):
    """Return the inverse of a quaternion batch-wise."""
    w, x, y, z = q.unbind(-1)
    norm_sq = w*w + x*x + y*y + z*z
    return torch.stack((w, -x, -y, -z), dim=-1) / norm_sq.unsqueeze(-1).clamp(min=1e-9)

@torch.compile
def q_rotate(q, v):
    """Rotate vector v by quaternion q batch-wise."""
    zeros = torch.zeros_like(v[..., :1])
    v_quat = torch.cat((zeros, v), dim=-1)
    q_v = q_mult(q, v_quat)
    q_v_q_inv = q_mult(q_v, q_inv(q))
    _, x, y, z = q_v_q_inv.unbind(-1)
    return torch.stack((x, y, z), dim=-1)

@torch.compile
def normalize_q(q, eps=1e-9):
    """Normalize quaternions batch-wise."""
    norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    return q / norm.clamp(min=eps)

def rot_to_quat(R):
    """Convert rotation matrix R to quaternion q batch-wise."""
    # R shape: (N, 3, 3)
    N = R.shape[0]
    q = torch.empty((N, 4), dtype=R.dtype, device=R.device)
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] # Trace
    is_large_trace = t > 0
    safe_t = torch.where(is_large_trace, t, torch.ones_like(t)) # Avoid sqrt of negative
    s_large = 0.5 / torch.sqrt(safe_t + 1.0).clamp(min=1e-9)

    # Compute for large trace cases
    q_large_trace = torch.stack([
        0.25 / s_large,
        (R[:, 2, 1] - R[:, 1, 2]) * s_large,
        (R[:, 0, 2] - R[:, 2, 0]) * s_large,
        (R[:, 1, 0] - R[:, 0, 1]) * s_large
    ], dim=-1)
    q = torch.where(is_large_trace.unsqueeze(-1), q_large_trace, q)

    # Compute for small trace cases
    is_small_trace = ~is_large_trace
    if is_small_trace.any():
        R_small = R[is_small_trace]
        small_trace_indices = torch.where(is_small_trace)[0]
        diag = torch.diagonal(R_small, dim1=-2, dim2=-1) # N_small, 3
        max_diag_idx = torch.argmax(diag, dim=-1) # N_small

        # Case where R[0,0] is max
        is_max_0 = (max_diag_idx == 0)
        if is_max_0.any():
            R_max_0 = R_small[is_max_0]
            q_idx = small_trace_indices[is_max_0]
            t_val = 1.0 + R_max_0[:, 0, 0] - R_max_0[:, 1, 1] - R_max_0[:, 2, 2]
            s_0 = torch.sqrt(t_val.clamp(min=1e-9)) * 2
            s_0_inv = 1.0 / s_0.clamp(min=1e-9)
            q[q_idx, 0] = (R_max_0[:, 2, 1] - R_max_0[:, 1, 2]) * s_0_inv # w
            q[q_idx, 1] = 0.25 * s_0 # x
            q[q_idx, 2] = (R_max_0[:, 0, 1] + R_max_0[:, 1, 0]) * s_0_inv # y
            q[q_idx, 3] = (R_max_0[:, 0, 2] + R_max_0[:, 2, 0]) * s_0_inv # z

        # Case where R[1,1] is max
        is_max_1 = (max_diag_idx == 1)
        if is_max_1.any():
            R_max_1 = R_small[is_max_1]
            q_idx = small_trace_indices[is_max_1]
            t_val = 1.0 + R_max_1[:, 1, 1] - R_max_1[:, 0, 0] - R_max_1[:, 2, 2]
            s_1 = torch.sqrt(t_val.clamp(min=1e-9)) * 2
            s_1_inv = 1.0 / s_1.clamp(min=1e-9)
            q[q_idx, 0] = (R_max_1[:, 0, 2] - R_max_1[:, 2, 0]) * s_1_inv # w
            q[q_idx, 1] = (R_max_1[:, 0, 1] + R_max_1[:, 1, 0]) * s_1_inv # x
            q[q_idx, 2] = 0.25 * s_1 # y
            q[q_idx, 3] = (R_max_1[:, 1, 2] + R_max_1[:, 2, 1]) * s_1_inv # z

        # Case where R[2,2] is max
        is_max_2 = (max_diag_idx == 2)
        if is_max_2.any():
            R_max_2 = R_small[is_max_2]
            q_idx = small_trace_indices[is_max_2]
            t_val = 1.0 + R_max_2[:, 2, 2] - R_max_2[:, 0, 0] - R_max_2[:, 1, 1]
            s_2 = torch.sqrt(t_val.clamp(min=1e-9)) * 2
            s_2_inv = 1.0 / s_2.clamp(min=1e-9)
            q[q_idx, 0] = (R_max_2[:, 1, 0] - R_max_2[:, 0, 1]) * s_2_inv # w
            q[q_idx, 1] = (R_max_2[:, 0, 2] + R_max_2[:, 2, 0]) * s_2_inv # x
            q[q_idx, 2] = (R_max_2[:, 1, 2] + R_max_2[:, 2, 1]) * s_2_inv # y
            q[q_idx, 3] = 0.25 * s_2 # z

    # Final check for NaNs/Infs which might occur in edge cases like identity matrix if not handled above
    identity_q_single = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q.device, dtype=q.dtype)
    q = torch.where(torch.isfinite(q).all(dim=-1, keepdim=True), q, identity_q_single)

    return normalize_q(q)


def quat_to_rot(q):
    """Convert quaternion q to rotation matrix R batch-wise."""
    # q shape: (N, 4) -> w, x, y, z
    q = normalize_q(q) # Ensure unit quaternion
    w, x, y, z = q.unbind(-1)
    N = q.size(0)
    R = torch.zeros((N, 3, 3), dtype=q.dtype, device=q.device)
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R[:, 0, 0] = 1.0 - 2.0 * (y2 + z2)
    R[:, 1, 1] = 1.0 - 2.0 * (x2 + z2)
    R[:, 2, 2] = 1.0 - 2.0 * (x2 + y2)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)
    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 2] = 2.0 * (yz - wx)
    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    return R


def direction_to_quat(direction_vector):
    """
    Converts a batch of 3D direction vectors to quaternions representing
    the rotation from the z-axis [0, 0, 1] to that direction vector.
    Handles vectors parallel or anti-parallel to the z-axis.

    Args:
        direction_vector (torch.Tensor): Batch of direction vectors (N, 3).
                                         Assumed to be normalized.

    Returns:
        torch.Tensor: Batch of quaternions (N, 4).
    """
    N = direction_vector.shape[0]
    device = direction_vector.device
    dtype = direction_vector.dtype

    # Reference vector (z-axis)
    ref_vec = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(N, -1)

    # Dot product to find the angle cosine
    dot_prod = torch.sum(ref_vec * direction_vector, dim=-1) # Shape (N,)
    angle = torch.acos(dot_prod.clamp(min=-1.0+1e-7, max=1.0-1e-7)) # Shape (N,)

    # Cross product to find the rotation axis
    axis = torch.cross(ref_vec, direction_vector, dim=-1) # Shape (N, 3)
    axis_norm = torch.linalg.norm(axis, dim=-1) # Shape (N,)

    # Handle cases where the direction is parallel or anti-parallel to ref_vec
    # Parallel case (dot_prod ~ 1): Identity rotation
    # Anti-parallel case (dot_prod ~ -1): 180-degree rotation around x-axis (arbitrary choice)
    is_parallel = (dot_prod > 1.0 - 1e-7)
    is_antiparallel = (dot_prod < -1.0 + 1e-7)

    # Default axis-angle representation (for non-collinear cases)
    safe_axis_norm = axis_norm.clamp(min=1e-9)
    axis_normalized = axis / safe_axis_norm.unsqueeze(-1)
    axis_angle = axis_normalized * angle.unsqueeze(-1)

    # Convert axis-angle to quaternion using the existing utility
    q_default = axis_angle_to_quat(axis_angle)

    # Create quaternions for parallel and anti-parallel cases
    q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).expand(N, -1)
    # 180 deg around x-axis: angle=pi, axis=[1,0,0] -> w=cos(pi/2)=0, xyz=[1,0,0]*sin(pi/2)=[1,0,0]
    q_180_x = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device, dtype=dtype).expand(N, -1)

    # Select the correct quaternion based on the case
    q = torch.where(is_parallel.unsqueeze(-1), q_identity, q_default)
    q = torch.where(is_antiparallel.unsqueeze(-1), q_180_x, q)

    return normalize_q(q) # Ensure normalization


def axis_angle_to_quat(axis_angle):
    """Convert axis-angle representation to quaternion batch-wise."""
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp(min=1e-9)
    angle_half = angle / 2.0
    w = torch.cos(angle_half)
    sin_angle_half = torch.sin(angle_half)
    xyz = axis * sin_angle_half
    return normalize_q(torch.cat([w, xyz], dim=-1))

# --- ORIGINAL FUNCTION (for reference or fallback) ---
def get_initial_quaternion_from_position(coords, edges, sigma=1.0):
    """
    Computes an initial local orthonormal basis using two types of neighbor-derived vectors,
    then converts to a quaternion.
    1. u_local1: Based on soft-weighted average neighbor direction (v_weighted_i).
    2. u_local2_aux: Based on unweighted average neighbor direction (v_unweighted_i)
                   or a fixed global axis if v_unweighted_i is degenerate or collinear with u_local1.
    The basis [b1, b2, b3] is constructed via Gram-Schmidt.
    Returns identity quaternion if u_local1 cannot be defined (e.g., isolated nodes).

    Args:
        coords (torch.Tensor): Node coordinates (position vectors $r_i$). Shape (N, 3).
        edges (torch.Tensor): Edge index (2, E), where edges[0] are source (j)
                              and edges[1] are target (i).
        sigma (float): Distance scaling hyperparameter for soft weighting of v_weighted_i.
                       Default: 1.0.

    Returns:
        torch.Tensor: Initial quaternions for each node. Shape (N, 4).
    """
    N, d = coords.shape
    assert d == 3, "Input coordinates must be 3D."
    device = coords.device
    dtype = coords.dtype
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).unsqueeze(0).expand(N, -1)
    eps = 1e-8

    if N == 0:
        return torch.empty((0, 4), device=device, dtype=dtype)

    v_weighted = torch.zeros_like(coords)
    v_unweighted = torch.zeros_like(coords)

    if edges.numel() > 0:
        row, col = edges # row = source (j), col = target (i)
        r_ji = coords[row] - coords[col] # Vector from i to j

        # 1. Calculate v_weighted (for u_local1)
        d_ij = torch.linalg.norm(r_ji, dim=-1).clamp(min=eps)
        sigma_val = max(float(sigma), eps)
        exp_term = torch.exp(-d_ij / sigma_val)
        sum_exp = scatter_add(exp_term, col, dim=0, dim_size=N).clamp(min=eps)
        w_ij = exp_term / sum_exp[col] # Weights for each edge targeting node i
        
        v_weighted = scatter_add(r_ji * w_ij.unsqueeze(-1), col, dim=0, dim_size=N)

        # 2. Calculate v_unweighted (for u_local2_aux)
        # Simply sum r_ji vectors for each node i. If you want average, divide by degree.
        # scatter_add effectively sums.
        v_unweighted = scatter_add(r_ji, col, dim=0, dim_size=N)

    # --- Primary Degeneracy Check: v_weighted is zero ---
    norm_v_weighted = torch.linalg.norm(v_weighted, dim=-1, keepdim=True)
    is_v_weighted_zero = (norm_v_weighted < eps) # Shape (N, 1)

    u_local1 = v_weighted / norm_v_weighted.clamp(min=eps) # Shape (N, 3)

    # --- Construct b2 using u_local2_aux or fallback global axis ---
    norm_v_unweighted = torch.linalg.norm(v_unweighted, dim=-1, keepdim=True)
    u_local2_aux_candidate = v_unweighted / norm_v_unweighted.clamp(min=eps) # Shape (N, 3)

    # Check if u_local2_aux_candidate is valid and not collinear with u_local1
    # Collinearity check: norm of cross product is close to zero
    cross_u1_u2aux = torch.cross(u_local1, u_local2_aux_candidate, dim=-1)
    norm_cross_u1_u2aux = torch.linalg.norm(cross_u1_u2aux, dim=-1, keepdim=True)

    # u_local2_aux is degenerate if v_unweighted was zero OR it's collinear with u_local1
    is_u2_aux_degenerate = (norm_v_unweighted < eps) | (norm_cross_u1_u2aux < eps) # Shape (N, 1)

    # Fallback: use a fixed global axis if u_local2_aux_candidate is degenerate
    dot_u1_z = u_local1[..., 2:3] # Keep dim for broadcasting, (N,1)
    is_u1_z_aligned = (torch.abs(dot_u1_z) > 1.0 - eps) # (N,1)

    # Global axes, expanded to (N,3) for torch.where
    global_z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(N, -1)
    global_y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand(N, -1)
    
    fallback_aux_axis = torch.where(is_u1_z_aligned, global_y_axis, global_z_axis) # (N,3)

    # Choose u_local2_aux: if degenerate, use fallback, else use candidate
    # Need to ensure dimensions match for torch.where
    # is_u2_aux_degenerate is (N,1), u_local2_aux_candidate (N,3), fallback_aux_axis (N,3)
    u_local2_aux = torch.where(is_u2_aux_degenerate, fallback_aux_axis, u_local2_aux_candidate)

    # Now, construct the orthonormal basis [b1, b2, b3]
    b1 = u_local1
    # b2 is perpendicular to b1 and u_local2_aux (or its fallback)
    # Recalculate cross product with the chosen u_local2_aux
    b2_unnormalized = torch.cross(b1, u_local2_aux, dim=-1)
    norm_b2 = torch.linalg.norm(b2_unnormalized, dim=-1, keepdim=True)
    
    # If norm_b2 is zero, it means b1 and u_local2_aux became collinear even after fallback.
    # This should ideally not happen if fallback logic is robust (e.g. b1 is not zero).
    # This can happen if b1 is zero, but that's covered by is_v_weighted_zero.
    # For extreme safety, handle if norm_b2 is zero by forcing b2 to something arbitrary
    # but this implies a deeper issue or an extremely specific input.
    # The most robust fallback for b2 if *everything* else fails and b1 is valid,
    # is to use the global axis method directly for b2 construction as in the previous answer.
    # Effectively, u_local2_aux selection already did this. So norm_b2 should not be zero if b1 is not zero.

    b2 = b2_unnormalized / norm_b2.clamp(min=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    # b3 should already be normalized. Can re-normalize for safety.
    # b3 = b3 / torch.linalg.norm(b3, dim=-1, keepdim=True).clamp(min=eps)

    calculated_basis = torch.stack([b1, b2, b3], dim=-1) # R = [b1, b2, b3] as columns
    calculated_quat = rot_to_quat(calculated_basis)

    # Apply identity quaternion for primary degenerate cases (v_weighted was zero)
    final_quat = torch.where(is_v_weighted_zero, identity_quat, calculated_quat)

    final_quat = torch.where(
        torch.isfinite(final_quat).all(dim=-1, keepdim=True),
        final_quat,
        identity_quat
    )
    return normalize_q(final_quat)

# --- NEW FUNCTION: PCA-based Initial Quaternion ---
def get_initial_quaternion_pca(coords):
    """
    Computes a single initial orientation based on the Principal Component Analysis (PCA)
    of the global node coordinates and assigns it to all nodes.
    The PCA axes form the initial local basis [pc1, pc2, pc3].
    Returns identity quaternions if N < 3.

    Args:
        coords (torch.Tensor): Node coordinates. Shape (N, 3).

    Returns:
        torch.Tensor: Initial quaternions for each node (all identical). Shape (N, 4).
    """
    N, d = coords.shape
    assert d == 3, "Input coordinates must be 3D."
    device = coords.device
    dtype = coords.dtype
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)

    # Handle edge case: PCA requires at least 3 points for 3 principal components.
    # Also handle N=0 case.
    if N < 3:
        print(f"Warning: Not enough nodes ({N}) for 3D PCA. Returning identity quaternions.")
        return identity_quat.unsqueeze(0).expand(N, -1)

    # 1. Center the coordinates (Subtract the mean)
    coords_mean = torch.mean(coords, dim=0, keepdim=True)
    coords_centered = coords - coords_mean

    # 2. Perform Singular Value Decomposition (SVD) on centered coordinates
    # SVD is often more numerically stable than computing the covariance matrix.
    # U * S @ Vh = coords_centered
    # The principal components are the rows of Vh (or columns of Vh.T).
    try:
        # Note: Need matrix of rank at least 3 for 3 distinct components.
        # If points are collinear or coplanar, SVD still works, but components might not be unique.
        U, S, Vh = torch.linalg.svd(coords_centered)
    except torch.linalg.LinAlgError as e:
         print(f"Warning: SVD failed ({e}). Returning identity quaternions.")
         return identity_quat.unsqueeze(0).expand(N, -1)


    # 3. Extract principal components (eigenvectors of covariance matrix)
    # Vh contains the principal axes as row vectors. We need them as column vectors
    # for the rotation matrix R = [pc1, pc2, pc3].
    R_pca = Vh.T # Shape (3, 3)

    # 4. Ensure a right-handed coordinate system (det(R) = +1)
    # SVD might return a reflection (det=-1). If so, flip the last principal component.
    if torch.linalg.det(R_pca) < 0:
        R_pca[:, 2] *= -1

    # 5. Convert the single PCA rotation matrix to a quaternion
    # rot_to_quat expects a batch dimension, so add and remove it.
    q_pca = rot_to_quat(R_pca.unsqueeze(0)).squeeze(0) # Shape (4,)

    # 6. Assign the same quaternion to all nodes
    initial_quats = q_pca.unsqueeze(0).expand(N, -1) # Shape (N, 4)

    # Final check for NaNs/Infs (shouldn't happen with SVD unless input was bad)
    initial_quats = torch.where(
        torch.isfinite(initial_quats).all(dim=-1, keepdim=True),
        initial_quats,
        identity_quat.unsqueeze(0).expand(N, -1) # Fallback to identity
    )

    return normalize_q(initial_quats) # Ensure normalization

class QFGNNLayer(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf):
        super().__init__()
        self.in_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.out_node_nf = out_node_nf
        self.in_edge_nf = in_edge_nf

        message_input_dim = 2 * in_node_nf + in_edge_nf + 1 + 4 + 3 # norm_r_ij(1), r_ij_dir_quat(4), q_rel_vec(3)
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_dim, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.LayerNorm(hidden_nf), # LayerNorm is typically applied *after* activation or block
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1),
            nn.Sigmoid() # Sigmoid activation for the gate
        )

        quat_update_input_dim = in_node_nf + hidden_nf
        self.quat_update_mlp = nn.Sequential(
            nn.Linear(quat_update_input_dim, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 3) # Predicts axis-angle update
        )

        node_update_input_dim = in_node_nf + hidden_nf
        self.node_update_mlp = nn.Sequential(
            nn.Linear(node_update_input_dim, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, out_node_nf)
        )

        self.alpha_h = nn.Parameter(torch.tensor(0.1))
        self.alpha_q = nn.Parameter(torch.tensor(0.1))

        # --- ADD Initialization ---
        self.init_weights()

    def init_weights(self):
        # Initialize message_mlp
        for i, layer in enumerate(self.message_mlp):
            if isinstance(layer, nn.Linear):
                # Use Kaiming for layers before SiLU
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu') # Use 'relu' heuristic for SiLU
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            # LayerNorm weights (gamma) and biases (beta) have default init (1s and 0s)

        # Initialize gate_mlp
        for i, layer in enumerate(self.gate_mlp):
            if isinstance(layer, nn.Linear):
                if i == len(self.gate_mlp) - 2: # Last linear layer before Sigmoid
                     nn.init.xavier_uniform_(layer.weight) # Xavier for Sigmoid
                else: # Layers before SiLU
                     nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Initialize quat_update_mlp
        for i, layer in enumerate(self.quat_update_mlp):
            if isinstance(layer, nn.Linear):
                if i == len(self.quat_update_mlp) - 1: # Last layer - Special Init
                    # Keep the original small gain initialization for stability
                    torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
                    torch.nn.init.zeros_(layer.bias)
                else: # Layer before SiLU
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Initialize node_update_mlp
        for i, layer in enumerate(self.node_update_mlp):
             if isinstance(layer, nn.Linear):
                 # Kaiming for all linear layers here, as they precede SiLU or are the output layer
                 nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                 if layer.bias is not None:
                     nn.init.zeros_(layer.bias)

        # Initialize alpha parameters (optional, could be learned or fixed)
        # nn.init.constant_(self.alpha_h, 0.1) # Already initialized in definition
        # nn.init.constant_(self.alpha_q, 0.1)


    def forward(self, h, q, x, edges, edge_attr):
        N = h.shape[0]
        row, col = edges[0], edges[1]
        target_idx, source_idx = col, row
        r_ij = x[source_idx] - x[target_idx]
        q_i = q[target_idx]
        norm_r_ij = torch.linalg.norm(r_ij, dim=-1, keepdim=True)
        safe_norm_r_ij = norm_r_ij.clamp(min=1e-9)
        r_ij_normalized = r_ij / safe_norm_r_ij
        r_hat_tilde_ij = q_rotate(q_inv(q_i), r_ij_normalized)
        r_ij_dir_quat = direction_to_quat(r_hat_tilde_ij) # Shape (E, 4)
        q_j = q[source_idx]
        q_rel = q_mult(q_inv(q_i), q_j)
        q_rel_vec = q_rel[:, 1:] # Shape (E, 3)
        h_i = h[target_idx]
        h_j = h[source_idx]

        # Ensure correct dimension: 2*h + edge + norm_r + dir_quat + q_rel_vec
        # Corrected calculation: 2*in_node_nf + in_edge_nf + 1 + 4 + 3
        # This matches the definition message_input_dim
        message_input = torch.cat([h_i, h_j, edge_attr, safe_norm_r_ij, r_ij_dir_quat, q_rel_vec], dim=-1)

        messages = self.message_mlp(message_input)
        gates = self.gate_mlp(messages)
        gated_messages = messages * gates
        aggregated_messages = scatter_add(gated_messages, target_idx, dim=0, dim_size=N)
        quat_update_input = torch.cat([h, aggregated_messages], dim=-1)
        axis_angle_update = self.alpha_q * self.quat_update_mlp(quat_update_input)
        delta_q = axis_angle_to_quat(axis_angle_update)
        q_new = q_mult(q, delta_q)
        q_new = normalize_q(q_new)
        node_update_input = torch.cat([h, aggregated_messages], dim=-1)
        h_delta = self.node_update_mlp(node_update_input)
        h_new = h + self.alpha_h * h_delta
        return h_new, q_new

# --- MODIFIED QuaternionFrameGNN with Initialization ---
class QuaternionFrameGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_final_nf, in_edge_nf, n_layers, initial_frame='local'):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_nf = hidden_nf
        self.initial_frame = initial_frame.lower()

        # Embedding layer
        self.node_embed = nn.Linear(in_node_nf, hidden_nf)
        current_node_nf = hidden_nf

        # Stack of QFGNN layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(QFGNNLayer( # Uses the modified layer with its own init
                in_node_nf=current_node_nf,
                hidden_nf=hidden_nf,
                out_node_nf=hidden_nf,
                in_edge_nf=in_edge_nf
            ))
            current_node_nf = hidden_nf # Assuming hidden_nf stays the same

        # Final MLP for output
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, out_final_nf)
        )

        if self.initial_frame not in ['pca', 'local']:
             raise ValueError(f"Invalid initial_frame: {initial_frame}. Choose 'pca' or 'local'.")

        # --- ADD Initialization ---
        self.init_weights()

    def init_weights(self):
         # Initialize node_embed
         nn.init.kaiming_uniform_(self.node_embed.weight, nonlinearity='relu')
         if self.node_embed.bias is not None:
            nn.init.zeros_(self.node_embed.bias)

         # Initialize final_mlp
         for i, layer in enumerate(self.final_mlp):
             if isinstance(layer, nn.Linear):
                 # Kaiming for all linear layers here
                 nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                 if layer.bias is not None:
                     nn.init.zeros_(layer.bias)

         # Note: self.layers (QFGNNLayer instances) are initialized within their own __init__

    def forward(self, h, x, edges, edge_attr):
        h_embed = self.node_embed(h)

        if self.initial_frame == 'pca':
             q = get_initial_quaternion_pca(x)
        elif self.initial_frame == 'local':
             q = get_initial_quaternion_from_position(x, edges, sigma=1.0)
        else:
             raise ValueError(f"Invalid initial_frame specified: {self.initial_frame}")

        current_h = h_embed
        current_q = q
        for layer in self.layers:
            current_h, current_q = layer(current_h, current_q, x, edges, edge_attr)

        final_features = self.final_mlp(current_h)

        # Optional reconstruction (unchanged)
        if final_features.shape[-1] == 3:
             final_local_basis = quat_to_rot(current_q)
             equivariant_coords_pred = torch.bmm(final_local_basis, final_features.unsqueeze(-1)).squeeze(-1)
             return equivariant_coords_pred
        else:
             return final_features


# --- QuaternionGNNBlock (If used separately, ensure QFGNNLayer is the modified one) ---
class QuaternionGNNBlock(nn.Module):
    """A block of multiple QFGNNLayers."""
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        current_nf = in_node_nf
        for i in range(n_layers):
            layer_out_nf = hidden_nf if i < n_layers - 1 else out_node_nf
            # Uses the QFGNNLayer defined above, which includes initialization
            self.layers.append(QFGNNLayer(
                in_node_nf=current_nf,
                hidden_nf=hidden_nf,
                out_node_nf=layer_out_nf,
                in_edge_nf=in_edge_nf,
            ))
            current_nf = layer_out_nf

    def forward(self, h, q, x, edges, edge_attr):
        current_h = h
        current_q = q
        for layer in self.layers:
            current_h, current_q = layer(current_h, current_q, x, edges, edge_attr)
        return current_h, current_q
