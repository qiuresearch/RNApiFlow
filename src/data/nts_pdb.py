#!/usr/bin/env python3

# Adapted from NuFold, Credit: https://github.com/kiharalab/NuFold

import torch
from src.data import data_transforms
from src.data import data_transform
from src.data import rigid_utils
from src.data.nts_constants import atom28_indices_by_name, NUM_NA_RESIDUE_ATOMS
from src.data import utils as du
import tree

import dataclasses
import io
from typing import Any, Mapping, Optional

from Bio.PDB import PDBParser
import numpy as np

from src.data import nts_constants
from src import utils

ilogger = utils.get_pylogger(__name__)

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class NucleicAcid:
    """Nucleic acid structure representation"""
    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are ...? .
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Nucleotide type for each base represented as an integer between 0 and 4,
    # where 4 is 'X'.
    restype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    def __get_item__(self, key: str) -> np.ndarray:
        """Allows for dict-like access to the attributes."""
        if not hasattr(self, key):
            raise KeyError(f'Key {key} not found in NucleicAcid.')
        return getattr(self, key)


def from_pdb_string(
    pdb_str: str,
    chain_id: Optional[str] = None
    ) -> NucleicAcid:
    """Takes a PDB string and constructs a NucleicAcid object.

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Args:
        pdb_str: The contents of the pdb file
        chain_id: If None, then the pdb file must contain a single chain (which
        will be parsed). If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
        A new `NucleicAcid` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
   
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f'Only single model PDBs are supported. Found {len(models)} models.'
        )
    model = models[0]

    atom_positions = []
    restype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != ' ':
                raise ValueError(
                    f'PDB contains an insertion code at chain {chain.id} and residue '
                    f'index {res.id[1]}. These are not supported.'
                )
            res_shortname = nts_constants.restypes_3to1.get(res.resname.strip(), 'X')
            restype_idx = nts_constants.restype_indices.get(
                res_shortname, nts_constants.restype_num
            )
            pos = np.zeros((nts_constants.atom28_names_num, 3))
            mask = np.zeros((nts_constants.atom28_names_num,))
            res_b_factors = np.zeros((nts_constants.atom28_names_num,))
            for atom in res:
                if atom.name not in nts_constants.atom28_names:
                    ilogger.debug(
                        f'Ignoring unknown atom {atom.name} in residue {res.resname} '
                        f'at chain {chain.id} and residue index {res.id[1]}.'
                    )
                    continue
                pos[nts_constants.atom28_indices_by_name[atom.name]] = atom.coord
                mask[nts_constants.atom28_indices_by_name[atom.name]] = 1.
                res_b_factors[nts_constants.atom28_indices_by_name[atom.name]] = atom.bfactor

            # If no known atom positions are reported for the residue then skip it.
            if np.sum(mask) < 0.5:
                ilogger.debug(
                    f'Skipping residue {res.resname} at chain {chain.id} and '
                    f'residue index {res.id[1]} because no known atoms are present.'
                )
                continue
            restype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)
    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return NucleicAcid(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        restype=np.array(restype),
        residue_index=np.array(residue_index),
        b_factors=np.array(b_factors)
    )


def nucleicacid_to_target_feats(nuclacid) -> dict:
    """
    Converts a NucleicAcid object to a dict of model-ready features, 
    identical to PDBNABaseDataset.__getitem__() from RNA-Frameflow.
    """
    # 1. Convert to torch tensors
    restype = torch.tensor(nuclacid.restype).long()
    atom_positions = torch.tensor(nuclacid.atom_positions).double()
    atom_mask = torch.tensor(nuclacid.atom_mask).double()
    num_res = restype.shape[0]

    # 2. Initialize chain features
    nts_feats = {
        "restype": restype,
        "all_atom_positions": atom_positions, #[:, :NUM_NA_RESIDUE_ATOMS],
        "all_atom_mask": atom_mask, #[:, :NUM_NA_RESIDUE_ATOMS],
        "atom_deoxy": torch.zeros(num_res, dtype=torch.bool),  # all RNA
    }

    # 3. Data transforms
    # nts_feats = data_transform.make_atom23_masks(nts_feats)
    # data_transform.atom23_list_to_atom28_list(
    #     nts_feats, ["all_atom_positions", "all_atom_mask"], inplace=True
    # )
    nts_feats = data_transform.atom28_to_frames(nts_feats)
    nts_feats = data_transform.atom28_to_torsion_angles()(nts_feats)

    # 4. Compute local frames
    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(
        nts_feats["rigidgroups_gt_frames"]
    )[:, 0]
    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()

    # 5. Backbone mask using C4′
    c4_idx = atom28_indices_by_name["C4'"]
    res_mask = (atom_mask[:, c4_idx] > 0).int()

    # 6. Final output dict
    final_feats = {
        "torsion_angles_sin_cos": nts_feats["torsion_angles_sin_cos"],
        "rotmats_1": rotmats_1,
        "trans_1": trans_1,
        "res_mask": res_mask,
        "is_na_residue_mask": torch.ones(num_res, dtype=torch.bool),
    }

    # 7. Padding
    final_feats = du.pad_feats(final_feats, num_res)
    
    # 8. Ensure everything is a tensor
    final_feats = tree.map_structure(
        lambda x: x if torch.is_tensor(x) else torch.tensor(x),
        final_feats
    )

    # 9. Float64 → Float32
    final_feats = {
        k: v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float64 else v
        for k, v in final_feats.items()
    }


    return final_feats

def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = 'TER'
    return (
        f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
        f'{chain_name:>1}{residue_index:>4}'
    )


def to_pdb(nuclacid: NucleicAcid) -> str:
    """Converts a `NucleicAcid` instance to a PDB string.
    Args:
      nuclacid: The protein to convert to PDB.
    Returns:
      PDB string.
    """
    restypes = nts_constants.restypes + ["X"]
    res_1to3 = lambda r: nts_constants.restypes_1to3.get(restypes[r], "UNK")

    pdb_lines = []

    atom_mask = nuclacid.atom_mask
    restype = nuclacid.restype
    atom_positions = nuclacid.atom_positions
    residue_index = nuclacid.residue_index.astype(np.int32)
    b_factors = nuclacid.b_factors

   #print("came into nucleicacid",restype)
    
    if np.any(restype > nts_constants.restype_num):
        raise ValueError("Invalid restypes.")

    atom_index = 1
    chain_id = "A"
    # Add all atom sites.
    
    for i in range(restype.shape[0]):
        res_name_3 = res_1to3(restype[i])
        for atom_name, pos, mask, b_factor in zip(
            nts_constants.atom28_names, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[
                0
            ]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            # print("atom_index",atom_index)
            # print("record_type",record_type)
            # print("name",name)
            # print("alt_loc",alt_loc)
            # print("residue_index[i]", residue_index[i], type(residue_index[i]))
            # print("pos", pos, pos.shape)
            # print("pos[0]", pos[0], type(pos[0]))
            # print("b_factor", b_factor, type(b_factor))

            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_id:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the chain.
    chain_end = "TER"
    chain_termination_line = (
        f"{chain_end:<6}{atom_index:>5}      {res_1to3(restype[-1]):>3} "
        f"{chain_id:>1}{residue_index[-1]:>4}"
    )
    pdb_lines.append(chain_termination_line)
    
    pdb_lines.append("END")
    pdb_lines.append("")
    return "\n".join(pdb_lines)

def from_prediction(
    features: FeatureDict,
    predicts: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = True,
) -> NucleicAcid:
    
    def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
        if remove_leading_feature_dimension and arr.ndim > 1 and arr.shape[0] == 1:
            return arr[0]  
        else:
            return arr
        
    if b_factors is None:
        b_factors = np.zeros_like(predicts["final_atom_mask"])
    

    return NucleicAcid(
        restype=_maybe_remove_leading_dim(features["restype"]),
        atom_positions=_maybe_remove_leading_dim(predicts["final_atom_positions"]),
        atom_mask=_maybe_remove_leading_dim(predicts["final_atom_mask"]),
        residue_index=_maybe_remove_leading_dim(features["residue_index"]) + 1,
        b_factors=_maybe_remove_leading_dim(b_factors),
    )



