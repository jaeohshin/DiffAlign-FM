from copy import deepcopy
import torch
from torch_geometric.data import Data
import rdkit
import rdkit.Chem.Draw
from rdkit import Chem
from rdkit.Chem import rdDepictor as DP
from rdkit.Chem import PeriodicTable as PT
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import Mol,GetPeriodicTable
from rdkit.Chem.Draw import rdMolDraw2D as MD2
from rdkit.Chem.rdmolops import RemoveHs
from typing import List, Tuple


BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BT.names.keys())}


def set_conformer_positions(conf, pos):
    for i in range(pos.shape[0]):
        conf.SetAtomPosition(i, pos[i].tolist())
    return conf

def update_data_rdmol_positions(data):
    for i in range(data.pos.size(0)):
        data.rdmol.GetConformer(0).SetAtomPosition(i, data.pos[i].tolist())
    return data


def update_data_pos_from_rdmol(data):
    new_pos = torch.FloatTensor(data.rdmol.GetConformer(0).GetPositions()).to(data.pos)
    data.pos = new_pos
    return data


def set_rdmol_positions(rdkit_mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = Chem.Mol(rdkit_mol)
    set_rdmol_positions_(mol, pos)
    return mol


def set_rdmol_positions_(mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def get_atom_symbol(atomic_number):
    return PT.GetElementSymbol(GetPeriodicTable(), atomic_number)


def mol_to_smiles(mol: Mol) -> str:
    return Chem.MolToSmiles(mol, allHsExplicit=True)


def mol_to_smiles_without_Hs(mol: Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def remove_duplicate_mols(molecules: List[Mol]) -> List[Mol]:
    unique_tuples: List[Tuple[str, Mol]] = []

    for molecule in molecules:
        duplicate = False
        smiles = mol_to_smiles(molecule)
        for unique_smiles, _ in unique_tuples:
            if smiles == unique_smiles:
                duplicate = True
                break

        if not duplicate:
            unique_tuples.append((smiles, molecule))

    return [mol for smiles, mol in unique_tuples]


def get_atoms_in_ring(mol):
    atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        for a in ring:
            atoms.add(a)
    return atoms


def get_2D_mol(mol):
    mol = deepcopy(mol)
    DP.Compute2DCoords(mol)
    return mol


def draw_mol_svg(mol,molSize=(450,150),kekulize=False):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        DP.Compute2DCoords(mc)
    drawer = MD2.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    # return svg.replace('svg:','')
    return svg


def get_best_rmsd(probe, ref):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)
    rmsd = MA.GetBestRMS(probe, ref)
    return rmsd


def bond_type_to_int(bond):
    bond_type = bond.GetBondType()
    if bond_type == Chem.rdchem.BondType.SINGLE:
        return 1
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        return 2
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        return 3
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        return 12
    else:
        assert "Bond type error"

def mol_to_graph_data_obj(mol):
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())
    x = torch.tensor(atom_features, dtype=torch.int)

    edges = []
    bond_types = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond_type_to_int(bond)

        edges.append((i, j))
        bond_types.append(bond_type)
        edges.append((j, i))
        bond_types.append(bond_type)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(bond_types, dtype=torch.float).view(-1)

    conf = mol.GetConformer()
    coordinates = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coordinates.append([pos.x, pos.y, pos.z])
    pos = torch.tensor(coordinates, dtype=torch.float)

    data = Data(atom_type=x, edge_index=edge_index, edge_type=edge_attr, pos=pos)
    
    return data
