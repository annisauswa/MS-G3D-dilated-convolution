import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def dilation_ratio(num_layers: int = 14) -> List[int]:
    return [2 ** (i % 7) for i in range(0, num_layers)]


def calculate_receptive_field(l_p_t: int, num_layers: int = 14) -> int:
    if l_p_t <= 0:
        return 1
    dr = dilation_ratio(num_layers)
    return sum(dr[:l_p_t]) + 1


class RecursiveLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation_rate = dilation_rate
        self.W1 = nn.Linear(in_channels, out_channels * 2, bias=False)
        self.W2 = nn.Linear(in_channels, out_channels * 2, bias=False)
        self.bias = nn.Parameter(torch.randn(out_channels * 2))
        self.norm = nn.LayerNorm(out_channels * 2)

        nn.init.xavier_uniform_(self.W1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.W2.weight, gain=0.1)

    def forward(self, C_t_minus_dl_l_minus_1, C_t_l_minus_1):
        sum_terms = self.W1(C_t_minus_dl_l_minus_1) + \
            self.W2(C_t_l_minus_1) + self.bias
        sum_terms = self.norm(sum_terms)
        out, gate = sum_terms.chunk(2, dim=-1)
        return out * torch.sigmoid(gate)


class DilatedConv1d(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 14, dilation_rates: Optional[List[int]] = None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dilation_rates = dilation_rates or dilation_ratio(num_layers)

        if len(self.dilation_rates) != num_layers:
            raise ValueError("dilation_rates length must equal num_layers")

        self.input_proj = nn.Linear(in_channels, hidden_channels, bias=True)
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.1)
        nn.init.zeros_(self.input_proj.bias)

        self.layers = nn.ModuleList()
        self.layers.append(RecursiveLayer(
            in_channels, hidden_channels, self.dilation_rates[0]))
        for i in range(1, num_layers):  
            self.layers.append(RecursiveLayer(
                hidden_channels, hidden_channels, self.dilation_rates[i]))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, T, C_in = x.shape
        device, dtype = x.device, x.dtype
        buffer = [None] * (self.num_layers + 1)
        buffer[0] = x

        projected = self.input_proj(
            x.reshape(B * T, C_in)).view(B, T, self.hidden_channels)

        for l in range(1, self.num_layers + 1):
            layer = self.layers[l - 1]
            d = self.dilation_rates[l - 1]
            out_ch = self.hidden_channels
            C_tl_seq = torch.zeros(B, T, out_ch, dtype=dtype, device=device)
            prev_seq = buffer[l - 1]

            for t in reversed(range(T)):
                C_t = prev_seq[:, t, :]
                C_t_d = prev_seq[:, t - d, :] if t - \
                    d >= 0 else torch.zeros_like(C_t)
                C_tl_seq[:, t, :] = layer(C_t_d, C_t)
            buffer[l] = C_tl_seq

        c_list = [projected.permute(0, 2, 1).contiguous()]
        for l in range(1, self.num_layers + 1):
            c_list.append(buffer[l].permute(0, 2, 1).contiguous())
        return c_list


class Model(nn.Module):
    def __init__(self, in_channels: int = 150, hidden_channels: int = 50, num_layers: int = 14, num_classes: int = 52):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dilated = DilatedConv1d(
            in_channels, hidden_channels, num_layers=num_layers)
        self.cls_fc1 = nn.Linear(hidden_channels, 50)
        self.cls_fc2 = nn.Linear(50, num_classes)
        self.reg_fc1 = nn.Linear(hidden_channels, 50)
        self.reg_fc2 = nn.Linear(50, 50)
        self.reg_fc3 = nn.Linear(50, 1)
        self.register_buffer("prev_s", torch.tensor(0.0))

    @staticmethod
    def select_layer_by_st(st_value: int, num_layers: int) -> int:
        target = int(st_value) + 1
        for l in range(1, num_layers + 1):
            rf = calculate_receptive_field(l, num_layers=num_layers)
            if rf >= target:
                return l
        return num_layers

    def classification_head(self, g_c_t: torch.Tensor) -> torch.Tensor:
        h = F.relu(g_c_t)
        h = F.relu(self.cls_fc1(h))
        logits = self.cls_fc2(h)
        return logits

    def regression_head(self, g_s_t: torch.Tensor) -> torch.Tensor:
        h = F.relu(g_s_t)
        h = F.relu(self.reg_fc1(h))
        h = F.relu(self.reg_fc2(h))
        s_pred = F.relu(self.reg_fc3(h))
        return s_pred

    def forward(self, x: torch.Tensor, st_prev: Optional[int] = None, distance: Optional[torch.Tensor] = None):
        B, T, C_in = x.shape
        # length num_layers+1, each (B, channels_l, T)
        c_list = self.dilated(x)
        c_t_l_list = [c[..., -1] for c in c_list]  # each (B, channels_l)

        # Normalize distance to prevent extreme receptive field selection
        if distance is not None:
            distance = torch.clamp(distance / 255.0, 0.0, 1.0)
            st_val = (distance * 255).long()
            # print('st_val distance:', st_val)
        else:
            st_val = torch.full((B,), int(self.prev_s.item()),
                                device=x.device, dtype=torch.long)

        g_c_t_list = []
        for b in range(B):
            l_p_t = self.select_layer_by_st(st_val[b], self.num_layers)
            if l_p_t >= 1:
                g_c_t = torch.stack([c_t_l_list[l][b]
                                    for l in range(1, l_p_t + 1)], dim=0).sum(dim=0)
            else:
                g_c_t = c_t_l_list[0][b]
            g_c_t_list.append(g_c_t)
        g_c_t = torch.stack(g_c_t_list, dim=0)

        g_s_t = torch.stack(c_t_l_list[1:], dim=0).sum(dim=0)
        g_c_t = torch.clamp(g_c_t, min=-10, max=10)
        g_s_t = torch.clamp(g_s_t, min=-10, max=10)

        logits = self.classification_head(g_c_t)
        s_pred = self.regression_head(g_s_t)
        s_pred = torch.tanh(s_pred)  # 🔥 bounded output for regression

        # update memory for eval mode
        if not self.training:
            self.prev_s.copy_(s_pred.mean().detach())

        return logits, s_pred


if __name__ == "__main__":
    # small smoke test
    B = 32
    C_in = 150
    T = 255
    model = Model(in_channels=C_in, hidden_channels=50,
                  num_layers=14, num_classes=51)
    x = torch.randn(B, T, C_in)
    # x = torch.randn(B, C_in, T)
    logits, s_pred = model(x)
    print("logits.shape:", logits.shape)  # (B, num_classes)
    print("s_pred.shape:", s_pred.shape)  # (B, 1)
