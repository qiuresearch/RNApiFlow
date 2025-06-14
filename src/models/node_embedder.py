"""
Neural network for embedding node features.

Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/models/node_embedder.py
"""

import torch
from torch import nn
from src.models import utils

class NodeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(NodeEmbedder, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        
        self.linear = nn.Linear(
            self._cfg.c_pos_emb + self._cfg.c_timestep_emb + 4, self.c_s)
        

    def embed_t(self, timesteps, mask):
        timestep_emb = utils.get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, timesteps, mask, onehot):
        # s: [b]

        b, num_res, device = mask.shape[0], mask.shape[1], mask.device

        # [b, n_res, c_pos_emb]
        pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = utils.get_index_embedding(
            pos, self.c_pos_emb, max_len=2056
        )
        pos_emb = pos_emb.repeat([b, 1, 1])
        pos_emb = pos_emb * mask.unsqueeze(-1)

        onehot = onehot.to(device)
        onehot = onehot * mask.unsqueeze(-1)
    
        # [b, n_res, c_timestep_emb]
        
        input_feats = [pos_emb]
        input_feats.append(onehot)

        # timesteps are between 0 and 1. Convert to integers.
        input_feats.append(self.embed_t(timesteps, mask))
        
        return self.linear(torch.cat(input_feats, dim=-1).float())