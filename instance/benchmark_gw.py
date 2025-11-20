# MAXCUT (Goemans-Williamson)

import glob
import re
import os
import sys

import cvxpy as cp
import numpy as np

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def compute_cut(sample, G_m):
    n_qubits = len(G_m)
    cut = 0
    for u in range(n_qubits):
        for v in range(u + 1, n_qubits):
            if sample[u] != sample[v]:
                cut += G_m[u, v]

    return cut


def safe_cholesky(X, eps=1e-8):
    # Symmetrize just in case of numerical asymmetry
    X = 0.5 * (X + X.T)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(X)

    # Clip tiny negatives to zero
    eigvals = np.clip(eigvals, 0, None)

    # Reconstruct PSD matrix
    X_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Now do Cholesky on the repaired PSD matrix
    return np.linalg.cholesky(X_psd + eps * np.eye(X.shape[0]))


def gw_sdp_maxcut(W):
    """
    Goemans-Williamson SDP relaxation for MaxCut, solved via CVXPY.
    Returns cut value and partition (approximate).
    """
    # Define SDP variable
    n = len(W)
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0, cp.diag(X) == 1]

    # Objective: maximize 1/4 sum_ij W_ij (1 - X_ij)
    obj = cp.Maximize(0.25 * cp.sum(cp.multiply(W, (1 - X))))
    prob = cp.Problem(obj, constraints)

    # Solve with SCS
    prob.solve(verbose=False)

    # Extract randomized rounding solution
    U = safe_cholesky(X.value)
    r = np.random.randn(U.shape[1])
    x = np.sign(U @ r)

    cut_value = compute_cut(x, W)

    sample = "".join(["1" if b < 0 else "0" for b in x])

    return sample, cut_value


if __name__ == "__main__":
    quality = int(sys.argv[1]) if len(sys.argv) > 1 else None
    repulsion_base = float(sys.argv[2]) if len(sys.argv) > 2 else None

    # Get the file path
    all_file =sorted(glob.glob("G*/G*.txt"), key=natural_keys)

    # Generate graph from file
    for i in range(len(all_file)):
        line_ct = 0
        n_nodes = 0
        with open(all_file[i]) as f:
            for line in f:
                if line_ct == 0: # Get graph size
                    n_nodes = int(line.split()[0])
                    graph = np.zeros((n_nodes, n_nodes), dtype=np.float32)
                else:
                    idx_i = int(line.split()[0]) - 1
                    idx_j = int(line.split()[1]) - 1
                    graph[idx_i, idx_j] = int(line.split()[2])
                line_ct += 1

        start = time.perf_counter()
        bitstring, cut_value = gw_sdp_maxcut(graph)
        end = time.perf_counter()

        print(f"{all_file[i].split('/')[0]}: {end - start} seconds, {cut_value}, {bitstring}")
