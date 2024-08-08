import os
import esm
from dataset_src.cath_imem_2nd import Cath_imem,dataset_argument
from torch.optim import Adam
from torch_geometric.data import Batch,Data
from dataset_src.utils import NormalizeProtein
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import torch.nn.functional as F
import torch
from tqdm import tqdm
amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

amino_acid_dict = {
                    0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I',
                    10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y',
                    19: 'V',
                    20: 'X', 21: 'X', 22: 'X', 23: 'X', 24: 'X', 25: 'X', 26: 'X', 27: 'X', 28: 'X',
                    29: 'X',
                    30: 'X', 31: 'X', 32: 'X', 33: 'X', 34: 'X', 35: 'X', 36: 'X', 37: 'X'
                }
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
    integer_encoded = []
    one_hot_list = torch.zeros(len(model_residues), len(struc_2nds_res_alphabet))

    current_position = 0
    for chain_id, residue_num in model_residues:
        dssp_key = (chain_id, (' ', residue_num, ' '))
        if (chain_id, residue_num) not in missing_residues and dssp_key in dssp:

            sec_structure_char = dssp[dssp_key][2]
            sec_structure_char = sec_structure_char.replace('-', 'L')
            integer_encoded.append(char_to_int[sec_structure_char])

            one_hot = F.one_hot(torch.tensor(integer_encoded[-1]), num_classes=8)
            one_hot_list[current_position] = one_hot
        else:
            print(pdb_filename, 'Missing residue: ', chain_id, residue_num, 'fill with 0')
        current_position += 1
    ss_encoding = one_hot_list[:current_position]
    return ss_encoding

def pdb2graph(dataset, filename, esmmodel, alphabet):
    rec, rec_coords, c_alpha_coords, n_coords, c_coords = dataset.get_receptor_inference(filename)
    struc_2nd_res = get_struc2ndRes(filename) #F.one_hot(torch.randint(0, 8, size=(len(rec_coords),1)).squeeze(), num_classes=8)
    rec_graph = dataset.get_calpha_graph(
                rec, c_alpha_coords, n_coords, c_coords, rec_coords, struc_2nd_res)
    if rec_graph:
        normalize_transform = NormalizeProtein(filename='../dataset/cath40_k10_imem_add2ndstrc/mean_attr.pt')
        graph = normalize_transform(rec_graph)
        with torch.no_grad():
            need_head_weights = False
            repr_layers = []
            return_contacts = False

            AA = torch.argmax(graph.x[:, :20], dim=1)

            AA_sequence = ''.join(amino_acid_dict[index.item()] for index in AA)

            x = torch.tensor(alphabet.encode(AA_sequence)).to(device)

            # 将得到的向量转换为形状 (1, T)
            tokens = x.unsqueeze(0)

            graph.original_x = F.one_hot(x, num_classes=33).cpu()

            assert tokens.ndim == 2
            padding_mask = tokens.eq(esmfold.padding_idx)  # B, T
            x = esmfold.embed_scale * esmfold.embed_tokens(tokens)
            if esmfold.token_dropout:
                x.masked_fill_((tokens == esmfold.mask_idx).unsqueeze(-1), 0.0)
                # x: B x T x C
                mask_ratio_train = 0.15 * 0.8
                src_lengths = (~padding_mask).sum(-1)
                mask_ratio_observed = (tokens == esmfold.mask_idx).sum(-1).to(x.dtype) / src_lengths
                x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

            if need_head_weights:
                attn_weights = []

            # (B, T, E) => (T, B, E)
            x = x.transpose(0, 1)

            if not padding_mask.any():
                padding_mask = None

            for layer_idx, layer in enumerate(esmfold.layers):
                x, attn = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_head_weights=need_head_weights,
                )

                if need_head_weights:
                    # (H, B, T, T) => (B, H, T, T)
                    attn_weights.append(attn.transpose(1, 0))

            x = esmfold.emb_layer_norm_after(x)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)
            graph.x = x.squeeze(0).cpu()
        return graph
    else:
        return None



if __name__ == '__main__':
    #### dataset  ####
    dataset_arg = dataset_argument(n=51)
    CATH_test_inmem = Cath_imem(dataset_arg['root'], dataset_arg['name'], split='test',
                                divide_num=dataset_arg['divide_num'], divide_idx=dataset_arg['divide_idx'],
                                c_alpha_max_neighbors=dataset_arg['c_alpha_max_neighbors'],
                                set_length=dataset_arg['set_length'],
                                struc_2nds_res_path=dataset_arg['struc_2nds_res_path'],
                                random_sampling=True, diffusion=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # # 设置新的路径
    # new_path = "/hy-tmp/conda/torch"
    # # 确保新的目录存在，如果不存在则创建
    # os.makedirs(new_path, exist_ok=True)
    #
    # # 设置环境变量
    # os.environ["TORCH_HOME"] = new_path
    # Load ESM-2 model
    esmfold, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esmfold = esmfold.to(device)
    batch_converter = alphabet.get_batch_converter()
    esmfold.eval()  # disables dropout for deterministic results
    error_pdb = []
    for key in ['test', 'validation', 'train']:
        pdb_dir = f'../dataset/raw/dompdb/'
        save_dir = f'../dataset/process_test/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename_list = [i for i in os.listdir(pdb_dir) if i.endswith('.pdb')]
        for filename in tqdm(filename_list):
            if os.path.exists(save_dir + filename.replace('.pdb', '.pt')):
                pass
            else:
                try:
                    graph = pdb2graph(CATH_test_inmem, pdb_dir + filename, esmfold, alphabet)
                    if graph:
                        torch.save(graph, save_dir + filename.replace('.pdb', '.pt'))
                    else:
                        error_pdb.append(filename)
                except (IndexError, KeyError):
                    error_pdb.append(filename)

    print(len(error_pdb))
    print(error_pdb)