# MAXCUT
# Produced by Dan Strano, Elara (the OpenAI custom GPT), and Gemini (Google Search AI)

# We reduce transverse field Ising model for globally uniform J and h parameters from a 2^n-dimensional problem to an (n+1)-dimensional approximation that suffers from no Trotter error. Upon noticing most time steps for Quantinuum's parameters had roughly a quarter to a third (or thereabouts) of their marginal probability in |0> state, it became obvious that transition to and from |0> state should dominate the mechanics. Further, the first transition tends to be to or from any state with Hamming weight of 1 (in other words, 1 bit set to 1 and the rest reset 0, or n bits set for Hamming weight of n). Further, on a torus, probability of all states with Hamming weight of 1 tends to be exactly symmetric. Assuming approximate symmetry in every respective Hamming weight, the requirement for the overall probability to converge to 1.0 or 100% in the limit of an infinite-dimensional Hilbert space suggests that Hamming weight marginal probability could be distributed like a geometric series. A small correction to exact symmetry should be made to favor closeness of "like" bits to "like" bits (that is, geometric closeness on the torus of "1" bits to "1" bits and "0" bits to "0" bits), but this does not affect average global magnetization. Adding an oscillation component with angular frequency proportional to J, we find excellent agreement with Trotterization approaching the limit of infinitesimal time step, for R^2 (coefficient of determination) of normalized marginal probability distribution of ideal Trotterized simulation as described by the (n+1)-dimensional approximate model, as well as for R^2 and RMSE (root-mean-square error) of global magnetization curve values.

# After tackling the case where parameters are uniform and independent of time, we generalize the model by averaging per-qubit behavior as if the static case and per-time-step behavior as finite difference. This provides the basis of a novel physics-inspired (adiabatic TFIM) MAXCUT approximate solver that often gives optimal or exact answers on a wide selection of graph types.

import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix
from pyqrackising import maxcut_tfim_sparse, spin_glass_solver_sparse

import glob
import re
import os
import sys


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def compute_cut(bitstring, G_m, n_qubits):
    sample = [b == '1' for b in list(bitstring)]
    cut = 0
    for u in range(n_qubits):
        for v in range(u + 1, n_qubits):
            if sample[u] != sample[v]:
                cut += G_m[u, v]

    return cut

if __name__ == "__main__":
    quality = int(sys.argv[1]) if len(sys.argv) > 1 else None
    repulsion_base = float(sys.argv[2]) if len(sys.argv) > 2 else None

    # Get the file path
    all_file =sorted(glob.glob("G*/G*.txt"), key=natural_keys)

    # Generate graph from file
    for i in range(len(all_file)):
        line_ct = 0
        n_nodes = 0
        min_weight = float("inf")
        max_weight = -float("inf")
        with open(all_file[i]) as f:
            for line in f:
                if line_ct == 0: # Get graph size
                    n_nodes = int(line.split()[0])
                    graph = lil_matrix((n_nodes, int(line.split()[0])), dtype=np.float32)
                else:
                    idx_i = int(line.split()[0]) - 1
                    idx_j = int(line.split()[1]) - 1
                    if idx_i > idx_j:
                        continue
                    weight = int(line.split()[2])
                    graph[idx_i, idx_j] = weight
                    if weight < min_weight:
                        min_weight = weight
                line_ct += 1

        graph = graph.tocsr()

        _graph = graph
        bitstring, cut_value, _ = maxcut_tfim_sparse(_graph, quality=quality, repulsion_base=repulsion_base, is_spin_glass=False)

        print(f"G{i + 1}: {cut_value}, {bitstring}")
