import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List

SPATIAL_DILATIONS = {1: 1, 2: 2, 3: 4}
OUTPUT_CHANNELS = 75
INPUT_CHANNELS = 3
NUM_JOINTS = 25

SKELETON_TREE: Dict[int, List[int]] = {
    0: [1, 12, 16],
    1: [20],
    2: [3],
    3: [],
    4: [5],
    5: [6],
    6: [7, 22],
    7: [21],
    8: [9],
    9: [10],
    10: [11, 24],
    11: [23],
    12: [13],
    13: [14],
    14: [15],
    15: [],
    16: [17],
    17: [18],
    18: [19],
    19: [],
    20: [2, 8, 4],
    21: [],
    22: [],
    23: [],
    24: []
}

class SSNetLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.SmoothL1Loss(reduction='mean')
        self.alpha = alpha

    def forward(self, logits, s_pred, class_labels, distance_targets, mask=None):
        cls_loss = self.ce_loss(logits, class_labels)
        s_pred = s_pred.squeeze(1)
        if mask is None or mask.sum() == 0:
            reg_loss = torch.tensor(0., device=logits.device)
        else:
            reg_loss = self.reg_loss_fn(s_pred[mask], distance_targets[mask])
        total_loss = cls_loss + self.alpha * reg_loss
        return total_loss, cls_loss, reg_loss


# class SSNetLoss(nn.Module):
#     def __init__(self, alpha: float = 1.0):
#         super().__init__()
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.mse_loss = nn.MSELoss()
#         self.alpha = alpha

#     def forward(self, logits, s_pred, class_labels, distance_targets):
#         cls_loss = self.ce_loss(logits, class_labels)
#         s_pred = s_pred.squeeze(1)  
#         mask = (class_labels != 0) 
#         if mask.sum() > 0:
#             reg_loss = self.mse_loss(s_pred[mask], distance_targets[mask])
#         else:
#             reg_loss = torch.tensor(0., device=s_pred.device)

#         total_loss = cls_loss + self.alpha * reg_loss

#         return total_loss, cls_loss, reg_loss


def GLU(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


class DilatedTreeConvolution:
    def __init__(self, layer_idx: int, input_dim: int, output_dim: int):
        self.d = SPATIAL_DILATIONS[layer_idx]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W_P = np.random.randn(output_dim, input_dim) * 0.1
        self.W_L = np.random.randn(output_dim, input_dim) * 0.1
        self.W_R = np.random.randn(output_dim, input_dim) * 0.1
        self.bias = np.random.randn(output_dim) * 0.1

    def _get_dilated_descendant(self, joint_idx: int, dilation: int) -> int:
        return joint_idx + dilation

    def _apply_zero_padding_and_get_inputs(self, joint_idx: int, input_features: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        C_parent = input_features.get(joint_idx, np.zeros(self.input_dim))
        C_left_descendant = np.zeros(self.input_dim)
        C_right_descendant = np.zeros(self.input_dim)
        children = SKELETON_TREE.get(joint_idx, [])

        if not children:
            pass
        elif len(children) == 1:
            child_idx = children
            dilated_descendant_idx = self._get_dilated_descendant(
                child_idx, self.d - 1)
            C_left_descendant = input_features.get(
                dilated_descendant_idx, np.zeros(self.input_dim))
        elif len(children) >= 2:
            left_child_idx = children
            right_child_idx = children[12]
            dilated_left_idx = self._get_dilated_descendant(
                left_child_idx, self.d - 1)
            dilated_right_idx = self._get_dilated_descendant(
                right_child_idx, self.d - 1)
            C_left_descendant = input_features.get(
                dilated_left_idx, np.zeros(self.input_dim))
            C_right_descendant = input_features.get(
                dilated_right_idx, np.zeros(self.input_dim))

        return C_parent, C_left_descendant, C_right_descendant

    def forward(self, input_features: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        output_activations = {}
        for joint_idx in range(1, NUM_JOINTS + 1):
            C_P, C_L, C_R = self._apply_zero_padding_and_get_inputs(
                joint_idx, input_features)
            conv_sum = np.dot(self.W_P, C_P) + np.dot(self.W_L,
                                                      C_L) + np.dot(self.W_R, C_R) + self.bias
            output_activations[joint_idx] = GLU(conv_sum)
        return output_activations


class SpatialRepresentationModule:
    def __init__(self):
        self.layer1 = DilatedTreeConvolution(
            1, INPUT_CHANNELS, OUTPUT_CHANNELS)
        self.layer2 = DilatedTreeConvolution(
            2, OUTPUT_CHANNELS, OUTPUT_CHANNELS)
        self.layer3 = DilatedTreeConvolution(
            3, OUTPUT_CHANNELS, OUTPUT_CHANNELS)

    def process_frame(self, raw_skeleton_input: Dict[int, np.ndarray]) -> np.ndarray:
        C_t_1 = self.layer1.forward(raw_skeleton_input)
        C_t_2 = self.layer2.forward(C_t_1)
        C_t_3 = self.layer3.forward(C_t_2)
        all_activations = []
        for features in [C_t_1, C_t_2, C_t_3]:
            all_activations.extend(list(features.values()))
        if not all_activations:
            return np.zeros(OUTPUT_CHANNELS)
        C_t_0 = np.mean(all_activations, axis=0)
        return C_t_0
