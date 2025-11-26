from unittest import result
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional


def dilation_rates(num_layers: int = 14) -> List[int]:
    return [2 ** (i % 7) for i in range(0, num_layers)]


def receptive_field(num_layers: int = 14) -> List[int]:
    dl = dilation_rates(num_layers)
    out = []
    s = 1   # base
    for i in range(num_layers):
        s += dl[i]
        out.append(s)
    return out

# def receptive_field(l_p_t:int) -> int:
#     return sum(dilation_rates()[:l_p_t]) + 1


class Conv1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, normalize: bool = False):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.W1 = nn.Linear(in_ch, out_ch * 2, bias=False)
        self.W2 = nn.Linear(in_ch, out_ch * 2, bias=False)
        self.bias = nn.Parameter(torch.rand(out_ch * 2))
        self.normalize = normalize
        if normalize:
            self.norm = nn.LayerNorm(out_ch * 2)
        init.xavier_uniform_(self.W1.weight, gain=1.0)
        init.xavier_uniform_(self.W2.weight,  gain=1.0)

    def forward(self, ctd: torch.Tensor, ctl: torch.Tensor) -> torch.Tensor:
        h = self.W1(ctd) + self.W2(ctl) + self.bias
        h = self.norm(h)
        u, v = h.chunk(2, dim=-1)
        out = u * torch.sigmoid(v)
        return out


class SSNet(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, num_layers: int = 14, num_classes: int = 52, pad_mode: str = "zero"):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.num_layers = num_layers
        self.dilation_rates = dilation_rates(num_layers)
        self.receptive_fields = receptive_field(num_layers)
        self.pad_mode = pad_mode
        layers = []
        layers.append(Conv1D(in_ch, hidden_ch, normalize=True))
        for _ in range(1, num_layers):
            layers.append(Conv1D(hidden_ch, hidden_ch, normalize=True))
        self.tcns = nn.ModuleList(layers)

        self.cls_fc1 = nn.Linear(hidden_ch, 50)
        self.cls_fc2 = nn.Linear(50, num_classes)
        self.reg_fc1 = nn.Linear(hidden_ch, 50)
        self.reg_fc2 = nn.Linear(50, 50)
        self.reg_fc3 = nn.Linear(50, 1)

    def _get_input_t(self, x: torch.Tensor, t: int):
        B, T, C = x.shape
        if 0 <= t < T:
            return x[:, t, :]
        else:
            if self.pad_mode == "zero":
                return x.new_zeros((B, C))
            else:
                return x[:, 0, :]

    def classification_head(self, g_c_t: torch.Tensor):
        h = F.relu(g_c_t)
        h = F.relu(self.cls_fc1(h))
        logits = self.cls_fc2(h)
        return logits

    def regression_head(self, g_s_t: torch.Tensor):
        h = F.relu(g_s_t)
        h = F.relu(self.reg_fc1(h))
        h = F.relu(self.reg_fc2(h))
        s_pred = F.relu(self.reg_fc3(h))
        return s_pred

    def select_layer_by_st(self, st_prev: torch.Tensor, num_layers: int = 14) -> torch.Tensor:
        B = st_prev.shape[0]
        lpt = torch.zeros((B,), dtype=torch.long, device=st_prev.device)
        for b in range(B):
            target = int(st_prev[b]) + 1
            for l in range(1, num_layers + 1):
                rf = self.receptive_fields[l - 1]
                if rf >= target:
                    lpt[b] = l
                    break
            else:
                lpt[b] = num_layers
        return lpt

    def forward(self, x: torch.Tensor, st_prev: torch.Tensor = None):
        B, T, C = x.shape
        t_final = T - 1
        memo: Dict[Tuple[int, int], torch.Tensor] = {}

        if st_prev is None or st_prev.shape[0] != B:
            st_prev = torch.zeros((B,), dtype=torch.long, device=x.device)
        else:
            if not isinstance(st_prev, torch.Tensor):
                st_prev = torch.tensor(
                    st_prev, device=x.device, dtype=torch.long)
            else:
                st_prev = st_prev.to(x.device)
            st_prev = st_prev.view(B,)

        def recurse(t: int, l: int) -> torch.Tensor:
            key = (t, l)
            if key in memo:
                return memo[key]

            dl = self.dilation_rates[l - 1]
            if l == 1:
                c_td = self._get_input_t(x, t - dl)
                c_t = self._get_input_t(x, t)
                out = self.tcns[0](c_td, c_t)
            else:
                c_td = recurse(t - dl, l - 1)
                c_t = recurse(t, l - 1)
                out = self.tcns[l - 1](c_td, c_t)

            memo[key] = out
            return out

        _ = recurse(t_final, self.num_layers)

        layer_outputs = []
        for l in range(1, self.num_layers + 1):
            key = (t_final, l)
            if key in memo:
                layer_outputs.append(memo[key].unsqueeze(1))
            else:
                layer_outputs.append(recurse(t_final, l).unsqueeze(1))

        result = torch.cat(layer_outputs, dim=1)

        lpt = self.select_layer_by_st(st_prev, self.num_layers)
        L = result.shape[1]
        H = result.shape[2]

        lpt = lpt.long().to(result.device)
        last_idx = (lpt - 1).clamp(min=0, max=L - 1)   # shape: [B]
        idx = torch.arange(L, device=result.device).unsqueeze(
            0)  # shape: [1, L]
        mask = (idx <= last_idx.unsqueeze(1)).unsqueeze(-1)  # shape: [B, L, 1]

        gct = (result * mask).sum(dim=1)  # shape: [B, H]
        gst = result.sum(dim=1)           # shape: [B, H]

        logits = self.classification_head(gct)   # shape: [B, num_classes]
        st_pred = self.regression_head(gst)      # shape: [B, 1]

        return logits, st_pred


if __name__ == "__main__":
    B, T, in_ch = 1000, 255, 150
    hidden_ch = 50
    num_layers = 14
    device = torch.device("cuda")
    x = torch.randn(B, T, in_ch)

    model = SSNet(in_ch=in_ch, hidden_ch=hidden_ch,
                  num_layers=num_layers, pad_mode="zero")
    model = model.to(device)
    x = x.to(device)
    outputs = model(x, st_prev=63)
    print("outputs.shape:", outputs.shape)
    assert outputs.shape == (B, num_layers, hidden_ch)
    print("layer 1 activation sample:", outputs[0, 0, :])
