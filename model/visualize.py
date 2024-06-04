import py3Dmol
from pymatgen.core.structure import Molecule 


def draw_mol_xyz(file_path):
    with open(file_path, 'r') as fo:
        xyz = fo.read()
    view = py3Dmol.view(width=800, height=400)
    view.addModel(xyz, 'xyz')
    view.setStyle({'sphere': {'scale': 0.35}, 'stick': {'radius': 0.20}})
    view.zoomTo()
    view.show()
    
    
def draw_mol_pymatgen(mol: Molecule):
    xyz_str = mol.to(fmt="xyz")
    view = py3Dmol.view(width=800, height=400)
    view.addModel(xyz_str, 'xyz')
    view.setStyle({'sphere': {'scale': 0.35}, 'stick': {'radius': 0.20}})
    view.zoomTo()
    view.show()
