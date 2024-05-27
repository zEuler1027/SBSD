import numpy as np
from pymatgen.core.periodic_table import Element
from pymatgen.core import Molecule
from pymatgen.io.xyz import XYZ
import torch
from typing import List


def atomic_numbers_to_symbols(atomic_numbers):
    return [Element.from_Z(number).symbol for number in atomic_numbers]


def write_batch_xyz(
    path_dir: str,
    atomic_numbers: torch.Tensor,
    mols_pos: torch.Tensor,
    mask: torch.Tensor
):
    node_strat = 0
    for i, mask_i in enumerate(mask):
        pos = mols_pos[node_strat: node_strat + mask_i]
        elements = atomic_numbers_to_symbols(atomic_numbers[node_strat: node_strat + mask_i])
        node_strat += mask_i
        mol = Molecule(
            np.array(elements),
            pos.cpu().numpy(),
        )
        xyz = XYZ(mol)
        xyz.write_file(f'{path_dir}/mol_{i}.xyz')
    assert node_strat == len(atomic_numbers)
