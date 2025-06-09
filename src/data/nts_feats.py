# Copyright 2022 Y.K, Kihara Lab
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
from typing import Union

from src.data.nts_constants import (
    restype_rigid_group_default_frame,
    restype_atom23_to_rigid_group,
    restype_atom23_mask,
    restype_atom23_rigid_group_positions,
)

from src.data.rigid_utils import Rotation, Rigid


def feats_to_atom23(trans, rots, restypes, torsions=None):

    backb_to_global = Rigid(
        Rotation(
            rot_mats=rots,
            quats=None
        ),
        trans,
    )

    all_frames_to_global = bp_torsion_angles_to_frames(
        backb_to_global,
        torsions,
        restypes,
    )

    pred_xyz = bp_frames_and_literature_positions_to_atom23_pos(
        all_frames_to_global,
        restypes,
    )

    return pred_xyz, all_frames_to_global.to_tensor_4x4()


def _init_residue_constants(float_dtype, device):
    
    default_frames = torch.tensor(
        restype_rigid_group_default_frame,
        dtype=float_dtype,
        device=device,
        requires_grad=False,
    )
    
    group_idx = torch.tensor(
        restype_atom23_to_rigid_group,
        device=device,
        requires_grad=False,
    )
    
    atom_mask = torch.tensor(
        restype_atom23_mask,
        dtype=float_dtype,
        device=device,
        requires_grad=False,
    )
    
    lit_positions = torch.tensor(
        restype_atom23_rigid_group_positions,
        dtype=float_dtype,
        device=device,
        requires_grad=False,
    )

    return default_frames, group_idx, atom_mask, lit_positions


def constant_to_tensor(
        constant: Union[torch.Tensor, float, list, tuple, int, np.ndarray],
        float_dtype: torch.dtype,
        device: torch.device,
        requires_grad: bool = False,):

    if constant is None:
        return None
    if isinstance(constant, torch.Tensor):
        return constant.to(dtype=float_dtype, device=device, requires_grad=requires_grad)
    else:
        return torch.tensor(
            constant,
            dtype=float_dtype,
            device=device,
            requires_grad=requires_grad,
        )


def get_tensor_default_frames(float_dtype, device):
    return constant_to_tensor(
        restype_rigid_group_default_frame,
        float_dtype,
        device,
    )


def bp_torsion_angles_to_frames(local_rigids, torsions, restypes):
    
    default_frames = get_tensor_default_frames(torsions.dtype, torsions.device)
    
    return torsion_angles_to_global_frames(local_rigids, torsions, restypes, default_frames)


def bp_frames_and_literature_positions_to_atom23_pos(global_rigids, restypes):

    default_frames, group_idx, atom_mask, lit_positions = _init_residue_constants(global_rigids.get_rots().dtype, global_rigids.get_rots().device)
    
    return frames_and_literature_positions_to_atom23_pos(
        global_rigids,
        restypes,
        default_frames,
        group_idx,
        atom_mask,
        lit_positions,
    )


def torsion_angles_to_global_frames(
    local_rigids: Rigid,
    torsions: torch.Tensor, # [*, N, T, 2] torsion angles, the last 2 dimensions are sin and cos
    restypes: torch.Tensor,
    rrgdf: torch.Tensor, # default rigid group frames for each residue type
):
    # [*, N, T, 4, 4]
    default_4x4 = rrgdf[restypes, ...]

    # 8 = T - 1?

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = local_rigids.from_tensor_4x4(default_4x4)

    # [*, 1, 1, 2] --> backbone to backbone rotation is identity
    b2b_rotation = torsions.new_zeros((*((1,) * len(torsions.shape[:-1])), 2))
    b2b_rotation[..., 1] = 1

    # [*, N, 8, 2]
    torsions = torch.cat(
        [b2b_rotation.expand(*torsions.shape[:-2], -1, -1), torsions], dim=-2
    )

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = torsions.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = torsions[..., 1]
    all_rots[..., 1, 2] = -torsions[..., 0]
    all_rots[..., 2, 1:] = torsions

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    g1_frame_to_bb = all_frames[..., 1]
    g2_frame_to_frame = all_frames[..., 2]
    g2_frame_to_bb = g1_frame_to_bb.compose(g2_frame_to_frame)
    g3_frame_to_frame = all_frames[..., 3]
    g3_frame_to_bb = g2_frame_to_bb.compose(g3_frame_to_frame)
    g4_frame_to_frame = all_frames[..., 4]
    g4_frame_to_bb = g3_frame_to_bb.compose(g4_frame_to_frame)
    g5_frame_to_frame = all_frames[..., 5]
    g5_frame_to_bb = g4_frame_to_bb.compose(g5_frame_to_frame)
    g6_frame_to_bb = all_frames[..., 6]
    g7_frame_to_bb = all_frames[..., 7]
    g8_frame_to_frame = all_frames[..., 8]
    g8_frame_to_bb = g7_frame_to_bb.compose(g8_frame_to_frame)
    g9_frame_to_bb = all_frames[..., 9]

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :1],
            g1_frame_to_bb.unsqueeze(-1),
            g2_frame_to_bb.unsqueeze(-1),
            g3_frame_to_bb.unsqueeze(-1),
            g4_frame_to_bb.unsqueeze(-1),
            g5_frame_to_bb.unsqueeze(-1),
            g6_frame_to_bb.unsqueeze(-1),
            g7_frame_to_bb.unsqueeze(-1),
            g8_frame_to_bb.unsqueeze(-1),
            g9_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = local_rigids[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom23_pos(  # was 14
    global_rigids: Rigid,
    restypes: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    # [*, N, 23, 4, 4]
    default_4x4 = default_frames[restypes, ...]

    # [*, N, 23]
    group_mask = group_idx[restypes, ...]

    # [*, N, 23, 8] (8 is the number of rigid groups per residue)
    group_mask = F.one_hot(
        group_mask.long(),  # somehow
        num_classes=default_frames.shape[-3], # 10
    )

    # [*, N, 23, 8]
    t_atoms_to_global = global_rigids[..., None, :] * group_mask

    # [*, N, 23]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    # [*, N, 23, 1]
    atom_mask = atom_mask[restypes, ...].unsqueeze(-1)

    # [*, N, 23, 3]
    lit_positions = lit_positions[restypes, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions