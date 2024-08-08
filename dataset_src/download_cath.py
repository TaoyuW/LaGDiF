import os
import json
from dataset_src.cath_imem_2nd import Cath_imem,dataset_argument
from torch.optim import Adam
from torch_geometric.data import Batch,Data
from dataset_src.utils import NormalizeProtein
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import torch.nn.functional as F
import torch
# os.system(f"wget -qnc -P ../dataset http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl")
# os.system(f"wget -qnc -P ../dataset http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json")
# os.system(f"wget -qnc -P ../dataset http://people.csail.mit.edu/ingraham/graph-protein-design/data/SPIN2/test_split_L100.json")
# os.system(f"wget -qnc -P ../dataset http://people.csail.mit.edu/ingraham/graph-protein-design/data/SPIN2/test_split_sc.json")
def get_struc2ndRes(pdb_filename):
    struc_2nds_res_alphabet = ['E', 'L', 'I', 'T', 'H', 'B', 'G', 'S']
    char_to_int = dict((c, i) for i, c in enumerate(struc_2nds_res_alphabet))
    p = PDBParser()
    structure = p.get_structure('', pdb_filename)
    model = structure[0]
    dssp = DSSP(model, pdb_filename, dssp='mkdssp')

    # From model, extract the list of amino acids
    model_residues = [(chain.id, residue.id[1]) for chain in model for residue in chain if residue.id[0] == ' ']
    # From DSSP, extract the list of amino acids
    dssp_residues = [(k[0], k[1][1]) for k in dssp.keys()]
    # Determine the missing amino acids
    missing_residues = set(model_residues) - set(dssp_residues)

    # Initialize a list of integers for known secondary structures,
    # and another list of zeroes for one-hot encoding
    ss_seqs = []
    integer_encoded = []
    one_hot_list = torch.zeros(len(model_residues), len(struc_2nds_res_alphabet))

    current_position = 0
    for chain_id, residue_num in model_residues:
        dssp_key = (chain_id, (' ', residue_num, ' '))
        if (chain_id, residue_num) not in missing_residues and dssp_key in dssp:

            sec_structure_char = dssp[dssp_key][2]
            sec_structure_char = sec_structure_char.replace('-', 'L')

            integer_encoded.append(char_to_int[sec_structure_char])
            ss_seqs.append(sec_structure_char)
            one_hot = F.one_hot(torch.tensor(integer_encoded[-1]), num_classes=8)
            one_hot_list[current_position] = one_hot
        else:
            print(pdb_filename, 'Missing residue: ', chain_id, residue_num, 'fill with 0')
        current_position += 1
    ss_encoding = one_hot_list[:current_position]
    return ss_encoding, ss_seqs

with open('chain_set_splits.json', 'r') as f:
    data = json.load(f)
    for key in data.keys():
        if key not in ['cath_nodes']:
            all_data_file = os.listdir('../dataset/cath40_k10_imem_add2ndstrc/raw/dompdb/')
            for pdb_code in data[key]:
                if pdb_code+'.pdb' in all_data_file and not os.path.exists('../dataset/SS/{}'.format(pdb_code)):
                    struc_2nd_res, ss_seqs = get_struc2ndRes('../dataset/cath40_k10_imem_add2ndstrc/raw/dompdb/' + pdb_code+'.pdb')
                    with open('../dataset/SS/{}'.format(pdb_code), 'w') as file:
                        file.write(''.join(ss_seqs))



