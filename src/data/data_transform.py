
#Partially from OpenFold
#Credit: https://github.com/aqlaboratory/openfold


import torch, os, sys
from src.data import nts_constants as rc
from Bio import SeqIO
import numpy as np
from functools import reduce, wraps
from src.data.rigid_utils import Rotation, Rigid

NUM_NTS = 4
NUM_ATOM23 = 23
NUM_ATOM28 = 28
NUM_AA = 20

def curry1(f):
    """Supply all arguments but the first."""
    @wraps(f)
    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    """ Adapted from RNAframeFlow data_transforms.py """
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def get_one_hot(fastaseq):
    
    nucleotide_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
    
    try:
        indices = [nucleotide_map[n] for n in fastaseq.upper()]
    except KeyError:
        raise ValueError("Unknown nucleotide in sequence. Only 'A', 'U', 'G', 'C' are allowed.")

    onehot = np.eye(4)[indices]
    
    return onehot


def sequence_to_tensor(sequence):
    
    nucleotide_map = {'A': 0, 'G': 1, 'C': 2, 'U': 3, 'X': 4}
    mapped_sequence = [nucleotide_map[nt] for nt in sequence]
    tensor_idx = torch.tensor(mapped_sequence, dtype=torch.long)
    
    return tensor_idx

def genMap(res_idx):

    restype_atom28_to_atom23 = []

    for rt in rc.restypes:
        atom_names = rc.atom23_names_by_resname[rc.restypes_1to3[rt]]
        
        atom_name_to_idx23 = {name: i for i, name in enumerate(atom_names)}
        
        restype_atom28_to_atom23.append(
            [
                (atom_name_to_idx23[name] if name in atom_name_to_idx23 else 0)
                for name in rc.atom28_names
            ]
        )

    restype_atom28_to_atom23.append([0] * 28)

    restype_atom28_to_atom23 = torch.tensor(
            restype_atom28_to_atom23,
            dtype=torch.int32,
        )

    residx_atom28_to_atom23 = restype_atom28_to_atom23[res_idx]

    return residx_atom28_to_atom23

def make_atom_mask(rna_target, input_dir):
    
    rnafasta = f"{input_dir}/{rna_target}/{rna_target}.fasta"

    if not os.path.exists(rnafasta):
        sys.exit(f"Couldn't locate the fasta file {rnafasta}, please provide a fasta file as input.")
    
    rnaseq = ""
    for record in SeqIO.parse(rnafasta, "fasta"):
        rnaseq = str(record.seq)
    
    res_idx = sequence_to_tensor(rnaseq)

    atom_map = genMap(res_idx)

    onehot = get_one_hot(rnaseq)
    
    return {
        "atom_map": atom_map,
        "restype": res_idx,
        "onehot": onehot
    }
    

def make_atom23_masks(nts):
    """Construct denser atom positions (23 dimensions instead of 28).
    Adapted from NuFold data_transform.py
    """
    restype_atom23_to_atom28 = []
    restype_atom28_to_atom23 = []
    restype_atom23_mask = []

    for rt in rc.restypes:
        atom_names = rc.atom23_names_by_resname[rc.restypes_1to3[rt]]
        restype_atom23_to_atom28.append(
            [(rc.atom28_indices_by_name[name] if name else 0) for name in atom_names]
        )
        atom_name_to_idx23 = {name: i for i, name in enumerate(atom_names)}
        restype_atom28_to_atom23.append(
            [
                (atom_name_to_idx23[name] if name in atom_name_to_idx23 else 0)
                for name in rc.atom28_names
            ]
        )

        restype_atom23_mask.append(
            [(1.0 if name else 0.0) for name in atom_names]
        )

    # Add dummy mapping for restype 'UNK'
    restype_atom23_to_atom28.append([0] * 23)
    restype_atom28_to_atom23.append([0] * 28)
    restype_atom23_mask.append([0.0] * 23)

    restype_atom23_to_atom28 = torch.tensor(
        restype_atom23_to_atom28,
        dtype=torch.int32,
        device=nts["restype"].device,
    )
    restype_atom28_to_atom23 = torch.tensor(
        restype_atom28_to_atom23,
        dtype=torch.int32,
        device=nts["restype"].device,
    )
    restype_atom23_mask = torch.tensor(
        restype_atom23_mask,
        dtype=torch.float32,
        device=nts["restype"].device,
    )
    nts_restypes = nts['restype'].to(torch.long)

    # create the mapping for (residx, atom23) --> atom28, i.e. an array
    # with shape (num_res, 23) containing the atom28 indices for this protein
    residx_atom23_to_atom28 = restype_atom23_to_atom28[nts_restypes]
    residx_atom23_mask = restype_atom23_mask[nts_restypes]

    nts["atom23_atom_exists"] = residx_atom23_mask
    nts["residx_atom23_to_atom28"] = residx_atom23_to_atom28.long()

    # create the gather indices for mapping back
    residx_atom28_to_atom23 = restype_atom28_to_atom23[nts_restypes]
    nts["residx_atom28_to_atom23"] = residx_atom28_to_atom23.long()

    # create the corresponding mask
    restype_atom28_mask = torch.zeros(
        [5, 28], dtype=torch.float32, device=nts["restype"].device
    )
    for restype, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restypes_1to3[restype_letter]
        atom_names = rc.atom_names_by_resname[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom28_indices_by_name[atom_name]
            restype_atom28_mask[restype, atom_type] = 1

    residx_atom28_mask = restype_atom28_mask[nts_restypes]
    nts["atom28_atom_exists"] = residx_atom28_mask

    return nts

def atom23_list_to_atom28_list(nts_feats, atom23_keys, inplace=False):
    """ Adapted from RNAframeFlow data_transforms.py """
    restypes = nts_feats["restype"].to(torch.long)
    assert (
        0 <= restypes.min() <= restypes.max() <= 4
    ), "Only nucleic acid residue inputs are allowed in `atom23_list_to_atom27_list()`."

    atom28_vals = []
    for key in atom23_keys:
        atom23_val = nts_feats[key]
        atom23_val = atom23_val.view(*atom23_val.shape[:2], -1)  # note: must be of shape [batch_size, num_nodes, -1]
        atom28_val = batched_gather(
            atom23_val,
            nts_feats["residx_atom28_to_atom23"],
            dim=-2,
            no_batch_dims=len(atom23_val.shape[:-2]),
        )

        atom28_val = (atom28_val * nts_feats["atom28_atom_exists"][..., None]).squeeze(-1)
        if inplace:
            nts_feats[key] = atom28_val
        atom28_vals.append(atom28_val)

    return atom28_vals



def atom28_to_frames(nts, eps=1e-8):
    """ Adapted from NuFold data_transforms.py """

    restypes: torch.Tensor = nts["restype"]
    all_atom_positions: torch.Tensor = nts["all_atom_positions"]
    all_atom_mask: torch.Tensor = nts["all_atom_mask"]

    batch_dims = len(restypes.shape[:-1])

    restype_rigidgroup_base_atom_names = np.full([5, 10, 3], "", dtype=object)
    restype_rigidgroup_base_atom_names[:, 0, :] = ["C2'", "C1'", "O4'"]  # sugar ring
    restype_rigidgroup_base_atom_names[:, 1, :] = ["C1'", "O4'", "C4'"]  # angle1
    restype_rigidgroup_base_atom_names[:, 2, :] = ["O4'", "C4'", "C5'"]  # angle2
    restype_rigidgroup_base_atom_names[:, 3, :] = ["C4'", "C5'", "O5'"]  # angle3
    restype_rigidgroup_base_atom_names[:, 4, :] = ["C5'", "O5'", "P"]  # angle4
    restype_rigidgroup_base_atom_names[:, 5, :] = ["O5'", "P", "OP1"]  # angle5
    restype_rigidgroup_base_atom_names[:, 6, :] = ["C1'", "C2'", "O2'"]  # angle6
    restype_rigidgroup_base_atom_names[:, 7, :] = ["C1'", "C2'", "C3'"]  # angle7
    restype_rigidgroup_base_atom_names[:, 8, :] = ["C2'", "C3'", "O3'"]  # angle8

    for restype, restype_letter in enumerate(rc.restypes):
        resname = rc.restypes_1to3[restype_letter]
        for chi_idx in range(1):
            if rc.chi_angles_mask[restype][chi_idx]:
                names = rc.chi_atoms_by_resname[resname][chi_idx]
                restype_rigidgroup_base_atom_names[
                    restype, 9, :
                ] = names[1:]

    restype_rigidgroup_mask = all_atom_mask.new_zeros(
        (*restypes.shape[:-1], 5, 10),
    )
    restype_rigidgroup_mask[..., 0:9] = 1
    restype_rigidgroup_mask[..., :4, 9:] = all_atom_mask.new_tensor(
        rc.chi_angles_mask
    )

    lookuptable = rc.atom28_indices_by_name.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    restype_rigidgroup_base_atom28_idx = lookup(
        restype_rigidgroup_base_atom_names,
    )
    restype_rigidgroup_base_atom28_idx = restypes.new_tensor(
        restype_rigidgroup_base_atom28_idx,
    )
    restype_rigidgroup_base_atom28_idx = (
        restype_rigidgroup_base_atom28_idx.view(
            *((1,) * batch_dims), *restype_rigidgroup_base_atom28_idx.shape
        )
    )

    residx_rigidgroup_base_atom28_idx = batched_gather(
        restype_rigidgroup_base_atom28_idx,
        restypes,
        dim=-3,
        no_batch_dims=batch_dims,
    )

    base_atom_pos = batched_gather(
        all_atom_positions,
        residx_rigidgroup_base_atom28_idx,
        dim=-2,
        no_batch_dims=len(all_atom_positions.shape[:-2]),
    )

    gt_frames = Rigid.from_3_points(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=eps,
    )

    group_exists = batched_gather(
        restype_rigidgroup_mask,
        restypes,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    gt_atoms_exist = batched_gather(
        all_atom_mask,
        residx_rigidgroup_base_atom28_idx,
        dim=-1,
        no_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=restypes.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 10, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1
    rots = Rotation(rot_mats=rots)

    gt_frames = gt_frames.compose(Rigid(rots, None))

    restype_rigidgroup_is_ambiguous = all_atom_mask.new_zeros(
        *((1,) * batch_dims), 5, 10
    )
    restype_rigidgroup_rots = torch.eye(
        3, dtype=all_atom_mask.dtype, device=restypes.device
    )
    restype_rigidgroup_rots = torch.tile(
        restype_rigidgroup_rots,
        (*((1,) * batch_dims), 5, 10, 1, 1),
    )

    for resname, _ in rc.atom_swaps_by_resname.items():
        restype = rc.restype_indices[rc.restypes_3to1[resname]]
        chi_idx = int(sum(rc.chi_angles_mask[restype]) - 1)
        restype_rigidgroup_is_ambiguous[..., restype, chi_idx + 9] = 1
        restype_rigidgroup_rots[..., restype, chi_idx + 9, 1, 1] = -1
        restype_rigidgroup_rots[..., restype, chi_idx + 9, 2, 2] = -1

    residx_rigidgroup_is_ambiguous = batched_gather(
        restype_rigidgroup_is_ambiguous,
        restypes,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = batched_gather(
        restype_rigidgroup_rots,
        restypes,
        dim=-4,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = Rotation(
        rot_mats=residx_rigidgroup_ambiguity_rot
    )
    alt_gt_frames = gt_frames.compose(
        Rigid(residx_rigidgroup_ambiguity_rot, None)
    )

    gt_frames_tensor = gt_frames.to_tensor_4x4()
    alt_gt_frames_tensor = alt_gt_frames.to_tensor_4x4()

    nts["rigidgroups_gt_frames"] = gt_frames_tensor
    nts["rigidgroups_gt_exists"] = gt_exists
    nts["rigidgroups_group_exists"] = group_exists
    nts["rigidgroups_group_is_ambiguous"] = residx_rigidgroup_is_ambiguous
    nts["rigidgroups_alt_gt_frames"] = alt_gt_frames_tensor

    return nts


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=4, chis=1, atoms=4]. The residue types are
      in the order specified in rc.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in rc.restypes:
        residue_name = rc.restypes_1to3[residue_name]
        residue_chi_angles = rc.chi_atoms_by_resname[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([rc.atom28_indices_by_name[atom] for atom in chi_angle])
        for _ in range(1 - len(atom_indices)):
            atom_indices.append(
                [0, 0, 0, 0]
            )  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 1)  # For UNKNOWN residue.

    return chi_atom_indices


@curry1
def atom28_to_torsion_angles(
    nts,
    prefix="",
):
    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)restype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 28, 3] atom positions (in atom28 format)
            * (prefix)all_atom_mask:
                [*, N_res, 28] atom position mask
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)alt_torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Alternate torsion angles (accounting for 180-degree symmetry)
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """
    restypes: torch.Tensor = nts[prefix + "restype"]
    all_atom_positions: torch.Tensor = nts[prefix + "all_atom_positions"]
    all_atom_mask: torch.Tensor = nts[prefix + "all_atom_mask"]

    restypes = torch.clamp(restypes, max=NUM_NTS)

    pad = all_atom_positions.new_zeros(
        [*all_atom_positions.shape[:-3], 1, NUM_ATOM28, 3]
    )  # [0 1 28 3]
    prev_all_atom_positions = torch.cat(
        [
            pad,
            all_atom_positions[..., :-1, :, :]
        ], dim=-3
    )  # [0 L 28 3]
    fol_all_atom_positions = torch.cat(
        [
            all_atom_positions[..., 1:, :, :],
            pad,

        ], dim=-3
    )  # [0 L 28 3]

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, NUM_ATOM28])  # [0 1 28]
    prev_all_atom_mask = torch.cat(
        [
            pad,
            all_atom_mask[..., :-1, :]
        ], dim=-2
    )  # [0 L 28]
    fol_all_atom_mask = torch.cat(
        [
            all_atom_mask[..., 1:, :],
            pad,
        ], dim=-2
    )  # [0 L 28]

    # main chain angles
    angle1_atom_pos = torch.cat(
        [
            all_atom_positions[..., 9:10, :], # C2'
            all_atom_positions[..., 11:12, :], # C1'
            all_atom_positions[..., 6:7, :], # O4'
            all_atom_positions[..., 5:6, :], # C4'
        ], dim=-2
    )
    angle1_mask = torch.prod(
        all_atom_mask[..., 9:10], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 11:12], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 6:7], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 5:6], dim=-1
    )

    angle2_atom_pos = torch.cat(
        [
            all_atom_positions[..., 11:12, :], # C1'
            all_atom_positions[..., 6:7, :], # O4'
            all_atom_positions[..., 5:6, :], # C4'
            all_atom_positions[..., 4:5, :], # C5'
        ], dim=-2
    )
    angle2_mask = torch.prod(
        all_atom_mask[..., 11:12], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 6:7], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 5:6], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 4:5], dim=-1
    )

    angle3_atom_pos = torch.cat(
        [
            all_atom_positions[..., 6:7, :], # O4'
            all_atom_positions[..., 5:6, :], # C4'
            all_atom_positions[..., 4:5, :], # C5'
            all_atom_positions[..., 3:4, :], # O5'
        ], dim=-2
    )
    angle3_mask = torch.prod(
        all_atom_mask[..., 6:7], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 5:6], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 4:5], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 3:4], dim=-1
    )

    angle4_atom_pos = torch.cat(
        [
            all_atom_positions[..., 5:6, :], # C4'
            all_atom_positions[..., 4:5, :], # C5'
            all_atom_positions[..., 3:4, :], # O5'
            all_atom_positions[..., 0:1, :], # P
        ], dim=-2
    )
    angle4_mask = torch.prod(
        all_atom_mask[..., 5:6], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 4:5], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 3:4], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 0:1], dim=-1
    )

    angle5_atom_pos = torch.cat(
        [
            all_atom_positions[..., 4:5, :], # C5'
            all_atom_positions[..., 3:4, :], # O5'
            all_atom_positions[..., 0:1, :], # P
            all_atom_positions[..., 1:2, :], # OP1
        ], dim=-2
    )
    angle5_mask = torch.prod(
        all_atom_mask[..., 4:5], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 3:4], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 0:1], dim=-1
    )  * torch.prod(
        all_atom_mask[..., 1:2], dim=-1
    )

    angle6_atom_pos = torch.cat(
        [
            all_atom_positions[..., 6:7, :], # O4'
            all_atom_positions[..., 11:12, :], # C1'
            all_atom_positions[..., 9:10, :], # C2'
            all_atom_positions[..., 10:11, :], # O2'
        ], dim=-2
    )
    angle6_mask = torch.prod(
        all_atom_mask[..., 6:7], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 11:12], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 9:10], dim=-1
    )  * torch.prod(
        all_atom_mask[..., 10:11], dim=-1
    )

    angle7_atom_pos = torch.cat(
        [
            all_atom_positions[..., 6:7, :], # O4'
            all_atom_positions[..., 11:12, :], # C1'
            all_atom_positions[..., 9:10, :], # C2'
            all_atom_positions[..., 7:8, :], # C3'
        ], dim=-2
    )
    angle7_mask = torch.prod(
        all_atom_mask[..., 6:7], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 11:12], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 9:10], dim=-1
    )  * torch.prod(
        all_atom_mask[..., 7:8], dim=-1
    )

    angle8_atom_pos = torch.cat(
        [
            all_atom_positions[..., 11:12, :], # C1'
            all_atom_positions[..., 9:10, :], # C2'
            all_atom_positions[..., 7:8, :], # C3'
            all_atom_positions[..., 8:9, :], # O3'
        ], dim=-2
    )
    angle8_mask = torch.prod(
        all_atom_mask[..., 11:12], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 9:10], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 7:8], dim=-1
    ) * torch.prod(
        all_atom_mask[..., 8:9], dim=-1
    )

    chi_atom_indices = torch.as_tensor(
        get_chi_atom_indices(), device=restypes.device
    )

    atom_indices = chi_atom_indices[..., restypes, :, :]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
    )

    chi_angles_mask = list(rc.chi_angles_mask)
    chi_angles_mask.append([0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[restypes, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
    )
    chis_mask = chis_mask * chi_angle_atoms_mask

    torsions_atom_pos = torch.cat(
        [
            angle1_atom_pos[..., None, :, :],
            angle2_atom_pos[..., None, :, :],
            angle3_atom_pos[..., None, :, :],
            angle4_atom_pos[..., None, :, :],
            angle5_atom_pos[..., None, :, :],
            angle6_atom_pos[..., None, :, :],
            angle7_atom_pos[..., None, :, :],
            angle8_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [
            angle1_mask[..., None],
            angle2_mask[..., None],
            angle3_mask[..., None],
            angle4_mask[..., None],
            angle5_mask[..., None],
            angle6_mask[..., None],
            angle7_mask[..., None],
            angle8_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(
        torsions_atom_pos[..., 3, :]
    )

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
            )
        + 1e-8
        )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom
    torsion_angles_sin_cos = torsion_angles_sin_cos * all_atom_mask.new_tensor(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # what is this ????
    )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]

    chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
        rc.chi_pi_periodic,
    )[restypes, ...]

    mirror_torsion_angles = torch.cat(
        [
            all_atom_mask.new_ones(*restypes.shape, 8),
            1.0 - 2.0 * chi_is_ambiguous,
        ],
        dim=-1,
    )
    # print("torsion_angles_sin_cos", torsion_angles_sin_cos.shape)
    # print("mirror_torsion_angles", mirror_torsion_angles.shape)
    alt_torsion_angles_sin_cos = (
        torsion_angles_sin_cos * mirror_torsion_angles[..., None]
    )

    nts[prefix + "torsion_angles_sin_cos"] = torsion_angles_sin_cos
    nts[prefix + "alt_torsion_angles_sin_cos"] = alt_torsion_angles_sin_cos
    nts[prefix + "torsion_angles_mask"] = torsion_angles_mask

    return nts


def get_backbone_frames(protein):
    # DISCREPANCY: AlphaFold uses tensor_7s here. I don't know why.
    protein["backbone_rigid_tensor"] = protein["rigidgroups_gt_frames"][
        ..., 0, :, :
    ]
    protein["backbone_rigid_mask"] = protein["rigidgroups_gt_exists"][..., 0]

    return protein


def get_chi_angles(protein):
    dtype = protein["all_atom_mask"].dtype
    # print(protein["torsion_angles_sin_cos"].shape)
    protein["chi_angles_sin_cos"] = (
        protein["torsion_angles_sin_cos"][..., 8:, :]
    ).to(dtype)
    protein["chi_mask"] = protein["torsion_angles_mask"][..., 8:].to(dtype)

    return protein