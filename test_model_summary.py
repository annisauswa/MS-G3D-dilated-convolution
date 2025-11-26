import torch
from torchinfo import summary
from model import msg3d, ssnet2
import graph
import io
import sys

# N, M, C, T, V = 1, 2, 3, 300, 25
B, T, C = 1, 255, 150

model_args = {
    # "num_class": 60,             # number of action classes
    # "num_point": V,              # number of joints
    # "num_person": M,             # number of persons in input
    # "num_gcn_scales": 13,        # number of GCN scales
    # "num_g3d_scales": 6,         # number of G3D scales
    # "graph": 'graph.ntu_rgb_d.AdjMatrixGraph',   # adjacency matrix graph
    "in_ch": 150,
    "hidden_ch": 50
}

model = ssnet2.Model(**model_args)

# dummy = torch.randn(N, C, T, V, M)
dummy = torch.rand(B, T, C)

# --- capture summary output ---
buffer = io.StringIO()
sys.stdout = buffer  # redirect stdout temporarily
summary(
    model,
    input_data=dummy,
    col_names=("input_size", "output_size", "num_params"),
    depth=4
)
sys.stdout = sys.__stdout__  # reset stdout

# --- write to file ---
with open("ssnet2_summary.txt", "w") as f:
    f.write(buffer.getvalue())

print("Model summary saved to ssnet2_summary.txt")
