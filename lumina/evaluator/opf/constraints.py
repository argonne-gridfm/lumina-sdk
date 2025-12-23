""" Constraints for ACOPF Problem.
Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

import numpy as np
import torch
from pandapower.pypower.idx_cost import (COST, MODEL, NCOST, POLYNOMIAL,
                                         PW_LINEAR, SHUTDOWN, STARTUP)


def compute_power_flow_violation_per_constraint(VM, VA, P_inj, Q_inj, Y_real, Y_imag):
    r"""Return per-bus power flow violations (constraint-wise).

    Computes the active/reactive power balance at each bus and returns the
    per-bus violation magnitude ``(P_balance**2 + Q_balance**2)``. Useful for
    reporting or reducing violations per constraint instead of an aggregated scalar.
    """
    VA_rad = torch.deg2rad(VA)

    # Calculate the voltage components
    V_real = VM * torch.cos(VA_rad)
    V_imag = VM * torch.sin(VA_rad)

    # Calculate the power flow (V_real + j V_imag) (Y_real + j Y_imag)
    P_flow = torch.matmul(V_real, Y_real) - torch.matmul(V_imag, Y_imag)
    Q_flow = torch.matmul(V_real, Y_imag) + torch.matmul(V_imag, Y_real)

    # Power balance per bus
    P_balance = P_inj - P_flow
    Q_balance = Q_inj - Q_flow

    return torch.abs(P_balance), torch.abs(Q_balance)


def compute_power_flow_violation(VM, VA, P_inj, Q_inj, Y_real, Y_imag, normalize=False):
    r"""Compute aggregated power flow violation using per-bus violations."""
    p_per_bus, q_per_bus = compute_power_flow_violation_per_constraint(
        VM=VM,
        VA=VA,
        P_inj=P_inj,
        Q_inj=Q_inj,
        Y_real=Y_real,
        Y_imag=Y_imag,
    )
    per_bus = p_per_bus**2 + q_per_bus**2
    if normalize:
        per_bus = per_bus / len(VM)
    return per_bus.sum()


def complex_csc_to_torch_sparse_csc(csc_mat):
    r"""Converts a complex scipy.sparse.csc_matrix to a torch.sparse_csc_tensor.

    Args:
        csc_mat (scipy.sparse.csc_matrix): The input complex CSC matrix.

    Returns:
        torch.sparse_csc_tensor: The equivalent complex sparse tensor.
    """
    if not np.iscomplexobj(csc_mat.data):
        raise ValueError("Input matrix must be complex.")

    # Extract real and imaginary parts
    real_data = csc_mat.data.real
    imag_data = csc_mat.data.imag

    # Extract CSC format components
    indices = torch.LongTensor(csc_mat.indices)
    indptr = torch.LongTensor(csc_mat.indptr)
    shape = csc_mat.shape

    # Create complex data tensor
    complex_data = torch.complex(torch.FloatTensor(real_data), torch.FloatTensor(imag_data))

    # Create sparse CSC tensor
    return torch.sparse_csc_tensor(indptr, indices, complex_data, shape)


def compute_power_flow_violation_v3(VM, VA, P_inj, Q_inj, Y_real, Y_imag, normalize=False):
    r"""Compute the power flow violation using real values. (bus loss) - batch version

    Args:
        VM (torch.Tensor): Voltage magnitudes.
        VA (torch.Tensor): Voltage angles in degrees.
        P_inj (torch.Tensor): Active power injections.
        Q_inj (torch.Tensor): Reactive power injections.
        Y_real (np.ndarray): Real part of admittance matrix.
        Y_imag (np.ndarray): Imaginary part of admittance matrix.
    Returns:
        torch.Tensor: Power flow violation.
    """
    # Convert angles from degrees to radians
    VA_rad = torch.deg2rad(VA)
    # Voltage complex
    V_comp = VM * torch.exp(1j * VA_rad)

    Y_real = create_block_diagonal_sparse_csc(Y_real, len(VM))
    Y_imag = create_block_diagonal_sparse_csc(Y_imag, len(VM))
    Y = Y_real + 1j * Y_imag
    S = Y @ V_comp
    P_flow = S.real
    Q_flow = S.imag
    if normalize:
        power_flow_violation = torch.norm(P_inj - P_flow)**2 + torch.norm(Q_inj - Q_flow)**2 / len(VM)
    else:
        power_flow_violation = (torch.norm(P_inj - P_flow)**2 + torch.norm(Q_inj - Q_flow)**2)
    return power_flow_violation


def compute_power_flow_violation_v2(VM, VA, P_inj, Q_inj, Y_real, Y_imag, normalize=False):
    r"""Compute the power flow violation using real values. (bus loss) - batch version

    Args:
        VM (torch.Tensor): Voltage magnitudes.
        VA (torch.Tensor): Voltage angles in degrees.
        P_inj (torch.Tensor): Active power injections.
        Q_inj (torch.Tensor): Reactive power injections.
        Y_real (np.ndarray): Real part of admittance matrix.
        Y_imag (np.ndarray): Imaginary part of admittance matrix.

    Returns:
        torch.Tensor: Power flow violation.
    """
    # Convert angles from degrees to radians
    VA_rad = torch.deg2rad(VA)

    # Calculate the voltage components
    V_real = VM * torch.cos(VA_rad)
    V_imag = VM * torch.sin(VA_rad)

    # NOTE: block format
    Y_real_block = torch.split(Y_real, Y_real.shape[1], dim=0)
    Y_imag_block = torch.split(Y_imag, Y_imag.shape[1], dim=0)
    Y_real = torch.block_diag(*Y_real_block)
    Y_imag = torch.block_diag(*Y_imag_block)

    # Calculate the power flow
    P_flow = torch.matmul(V_real, Y_real) - torch.matmul(V_imag, Y_imag)
    Q_flow = torch.matmul(V_real, Y_imag) + torch.matmul(V_imag, Y_real)

    # Calculate the power balance
    P_balance = P_inj - P_flow
    Q_balance = Q_inj - Q_flow

    # Calculate the power flow violation
    # due the batch loading of the data
    if normalize:
        power_flow_violation = torch.norm(P_balance, p=2) + torch.norm(Q_balance, p=2)
    else:
        power_flow_violation = (torch.norm(P_balance, p=2) + torch.norm(Q_balance, p=2)) / len(VM)

    return power_flow_violation


def compute_generation_cost(gen_cost, pg, normalize=False):
    r"""Compute the generation cost for a given generator output. (gen loss)

    Args:
        gen_cost (torch.Tensor): Generator cost coefficients.
        pg (torch.Tensor): Generator active power output.

    Returns:
        torch.Tensor: Computed generation cost.
    """
    cost = torch.zeros(1, device=pg.device, requires_grad=True)
    for i in range(len(pg)):

        if gen_cost[i, MODEL] == POLYNOMIAL:
            # Quadratic cost function
            num_coeffs = int(gen_cost[i, NCOST])
            for j in range(num_coeffs):
                coeff = gen_cost[i, COST + j]
                cost = cost + coeff * pg[i]**(num_coeffs - j - 1)

        elif gen_cost[i, MODEL] == PW_LINEAR:
            # Piecewise linear cost function
            n_segments = int(gen_cost[i, NCOST])
            segments = gen_cost[i, COST:COST + 2 * n_segments].reshape(n_segments, 2)
            for j in range(n_segments):
                if pg[i] <= segments[j, 0]:
                    cost = cost + segments[j, 1] * pg[i]
                    break
                else:
                    cost = cost + segments[j, 1] * segments[j, 0]
                    pg[i] -= segments[j, 0]
    if normalize:
        return cost.sum() / len(pg)
    else:
        return cost.sum()


def compute_line_limit_violation(VM, VA, Y_real, Y_imag, line_limits, edge_index, normalize=False):
    r"""Compute the line limit violation. (line loss)

    Args:
        VM (torch.Tensor): Voltage magnitudes.
        VA (torch.Tensor): Voltage angles in degrees.
        Y_real (np.ndarray): Real part of admittance matrix.
        Y_imag (np.ndarray): Imaginary part of admittance matrix.
        line_limits (torch.Tensor): Line flow limits.
        edge_index (torch.Tensor): Edge index tensor.

    Returns:
        torch.Tensor: Line limit violation.
    """
    # Convert angles from degrees to radians
    VA_rad = torch.deg2rad(VA)

    # Compute real and imaginary parts of bus voltages
    V_real = VM * torch.cos(VA_rad)
    V_imag = VM * torch.sin(VA_rad)

    # Compute real and imaginary parts of bus currents: I = Y * V
    # NOTE: I_real, I_imag are not used in the calculation
    # I_real = torch.matmul(Y_real, V_real) - torch.matmul(Y_imag, V_imag)
    # I_imag = torch.matmul(Y_real, V_imag) + torch.matmul(Y_imag, V_real)

    # Initialize the power flow tensors
    Viol_from = torch.zeros(line_limits.size(), device=VM.device)
    Viol_to = torch.zeros(line_limits.size(), device=VM.device)

    for k in range(len(edge_index[0])):
        i = edge_index[0, k]
        j = edge_index[1, k]

        # Compute branch current components
        I_branch_real = torch.matmul(Y_real[i, :] - Y_real[j, :], V_real) - \
            torch.matmul(Y_imag[i, :] - Y_imag[j, :], V_imag)
        I_branch_imag = torch.matmul(Y_real[i, :] - Y_real[j, :], V_imag) + \
            torch.matmul(Y_imag[i, :] - Y_imag[j, :], V_real)

        # Compute power flow at 'from' bus
        P_from = V_real[i] * I_branch_real + V_imag[i] * I_branch_imag
        Q_from = V_real[i] * I_branch_imag - V_imag[i] * I_branch_real

        # Compute power flow at 'to' bus
        P_to = V_real[j] * I_branch_real + V_imag[j] * I_branch_imag
        Q_to = V_real[j] * I_branch_imag - V_imag[j] * I_branch_real

        # Compute line limit violation
        Viol_from[k] = P_from**2 + Q_from**2 - VM[i]**2 * line_limits[k]**2
        Viol_to[k] = P_to**2 + Q_to**2 - VM[j]**2 * line_limits[k]**2

    # Calculate the line limit violation
    # line_limit_violation = torch.relu(Viol_from).sum() + torch.relu(Viol_to).sum()
    line_limit_violation = torch.max(torch.relu(Viol_from).max(), torch.relu(Viol_to).max())

    if normalize:
        return line_limit_violation / len(edge_index[0])
    else:
        return line_limit_violation


def compute_line_limit_violation_v2(VM, VA, Y_real, Y_imag, line_limits, edge_index, normalize=False):
    r"""Compute the line limit violation. (line loss)

    Args:
        VM (torch.Tensor): Voltage magnitudes.
        VA (torch.Tensor): Voltage angles in degrees.
        Y_real (np.ndarray): Real part of admittance matrix.
        Y_imag (np.ndarray): Imaginary part of admittance matrix.
        line_limits (torch.Tensor): Line flow limits.
        edge_index (torch.Tensor): Edge index tensor.

    Returns:
        torch.Tensor: Line limit violation.
    """
    # Convert angles from degrees to radians
    VA_rad = torch.deg2rad(VA)

    # Compute real and imaginary parts of bus voltages
    V_real = VM * torch.cos(VA_rad)
    V_imag = VM * torch.sin(VA_rad)

    # Compute real and imaginary parts of bus currents: I = Y * V
    # NOTE: I_real, I_imag are not used in the calculation
    # I_real = torch.matmul(Y_real, V_real) - torch.matmul(Y_imag, V_imag)
    # I_imag = torch.matmul(Y_real, V_imag) + torch.matmul(Y_imag, V_real)

    # Initialize the power flow tensors
    Viol_from = torch.zeros(line_limits.size(), device=VM.device)
    Viol_to = torch.zeros(line_limits.size(), device=VM.device)

    # NOTE: block format
    Y_real_block = torch.split(Y_real, Y_real.shape[1], dim=0)
    Y_imag_block = torch.split(Y_imag, Y_imag.shape[1], dim=0)
    Y_real = torch.block_diag(*Y_real_block)
    Y_imag = torch.block_diag(*Y_imag_block)

    for k in range(len(edge_index[0])):
        i = edge_index[0, k]
        j = edge_index[1, k]

        # Compute branch current components
        I_branch_real = torch.matmul(Y_real[i, :] - Y_real[j, :], V_real) - \
            torch.matmul(Y_imag[i, :] - Y_imag[j, :], V_imag)
        I_branch_imag = torch.matmul(Y_real[i, :] - Y_real[j, :], V_imag) + \
            torch.matmul(Y_imag[i, :] - Y_imag[j, :], V_real)

        # Compute power flow at 'from' bus
        P_from = V_real[i] * I_branch_real + V_imag[i] * I_branch_imag
        Q_from = V_real[i] * I_branch_imag - V_imag[i] * I_branch_real

        # Compute power flow at 'to' bus
        P_to = V_real[j] * I_branch_real + V_imag[j] * I_branch_imag
        Q_to = V_real[j] * I_branch_imag - V_imag[j] * I_branch_real

        # Compute line limit violation
        Viol_from[k] = P_from**2 + Q_from**2 - VM[i]**2 * line_limits[k]**2
        Viol_to[k] = P_to**2 + Q_to**2 - VM[j]**2 * line_limits[k]**2

    # Calculate the line limit violation
    line_limit_violation = torch.relu(Viol_from).sum() + torch.relu(Viol_to).sum()

    if normalize:
        return line_limit_violation / len(edge_index[0])
    else:
        return line_limit_violation


def convert_to_torch_sparse(sp_mat):
    r"""Convert a scipy sparse matrix to a torch sparse tensor.

    Args:
        sp_mat (scipy.sparse.csc_matrix): The input sparse sp_mat.

    Returns:
        torch.sparse.FloatTensor: The converted sparse tensor.
    """
    import scipy.sparse as sp
    if not isinstance(sp_mat, sp.spmatrix):
        raise ValueError("Input matrix must be a scipy sparse matrix.")

    if isinstance(sp_mat, sp.csc_matrix):
        indptr = sp_mat.indptr
        indices = sp_mat.indices
        values = sp_mat.data
        sp_pt = torch.sparse_csc_tensor(indptr, indices, values, sp_mat.shape)
    elif isinstance(sp_mat, sp.csr_matrix):
        raise NotImplementedError("CSR format is not supported yet.")

    return sp_pt


def create_block_diagonal_sparse_csc(matrix, n_blocks, format="csc"):
    r"""Create a block diagonal sparse CSC tensor with n_blocks copies of CSC tensor

    Args:
        matrix (torch.sparse_csc_tensor): The matrix to use as blocks
        n_blocks (int): Number of blocks to create

    Returns:
        torch.sparse_csc_tensor: Block diagonal sparse tensor
    """
    # Get dimensions and components of the original matrix
    m, n = matrix.shape
    assert m == n, "Matrix must be square"

    if format == "csc":
        # Get the original CSC components
        ccol_indices = matrix.ccol_indices()
        row_indices = matrix.row_indices()
        values = matrix.values()
        nnz_per_block = len(row_indices)

        # Create arrays for the new tensor
        new_size = n * n_blocks
        new_nnz = nnz_per_block * n_blocks

        # Initialize new CSC components
        new_ccol_indices = torch.zeros(new_size + 1, dtype=torch.int64)
        new_row_indices = torch.zeros(new_nnz, dtype=torch.int64)
        new_values = torch.zeros(new_nnz, dtype=values.dtype)

        # Fill the components block by block
        for i in range(n_blocks):
            # Calculate offsets for this block
            offset_idx = i * nnz_per_block
            offset_dim = i * n

            # Copy and adjust row indices
            new_row_indices[offset_idx:offset_idx + nnz_per_block] = row_indices + offset_dim

            # Copy values directly
            new_values[offset_idx:offset_idx + nnz_per_block] = values

            # Adjust ccol_indices
            new_ccol_indices[offset_dim:offset_dim + n + 1] = ccol_indices + (i * nnz_per_block)

        # Fix ccol_indices to ensure it's monotonically increasing
        for i in range(1, len(new_ccol_indices)):
            if new_ccol_indices[i] < new_ccol_indices[i - 1]:
                new_ccol_indices[i:] = new_ccol_indices[i - 1]

        # Create the new sparse CSC tensor
        block_diag = torch.sparse_csc_tensor(
            new_ccol_indices,
            new_row_indices,
            new_values,
            (new_size, new_size)
        )
    elif format == "coo":
        # Convert to COO format for easier manipulation
        # Extract CSC components and convert to COO
        if matrix.layout == torch.sparse_csc:
            matrix_coo = matrix.to_sparse_coo()
        else:
            matrix_coo = matrix  # Already in COO format

        indices = matrix_coo.indices()
        values = matrix_coo.values()

        # Create new indices and values for block diagonal structure
        new_indices = []
        new_values = []

        # For each block, add the indices and values with appropriate offsets
        for i in range(n_blocks):
            offset = i * n

            # Copy indices with offset
            block_indices = indices.clone()
            block_indices[0, :] += offset  # Row indices
            block_indices[1, :] += offset  # Column indices

            new_indices.append(block_indices)
            new_values.append(values)

        # Concatenate all blocks
        new_indices = torch.cat(new_indices, dim=1)
        new_values = torch.cat(new_values)

        # Create new sparse tensor
        new_size = (n * n_blocks, n * n_blocks)
        block_diag = torch.sparse_coo_tensor(new_indices, new_values, new_size)

        # Convert back to CSC if needed
        if matrix.layout == torch.sparse_csc:
            block_diag = block_diag.to_sparse_csc()
    else:
        raise ValueError("Unsupported format. Use 'csc' or 'coo'.")
    return block_diag
