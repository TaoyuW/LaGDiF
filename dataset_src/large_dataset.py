
from torch_geometric.data import Dataset, download_url,Batch,Data
import torch
import os
from torch_geometric.loader import DataListLoader, DataLoader
import random

class Cath(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, baseDIR,transform=None, pre_transform=None, pre_filter=None,pred_sasa = False,
                 hidden_dim: int = 320, latent_dim: int = 320, edge_dim: int = 93, aa_num: int = 20):
        super().__init__(baseDIR, transform, pre_transform, pre_filter)
        'Initialization'
        self.list_IDs = list_IDs
        self.baseDIR = baseDIR
        self.pred_sasa = pred_sasa
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.edge_dim = edge_dim
        self.aa_num = aa_num

    def len(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def get(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        data=torch.load(self.baseDIR+ID)

        del data['distances']
        del data['edge_dist']
        mu_r_norm=data.mu_r_norm
        if self.pred_sasa:
            extra_x_feature = torch.cat([data.x[:, 22:], mu_r_norm], dim=1)
        else:
            extra_x_feature = torch.cat([data.x[:, 20].unsqueeze(dim=1), data.x[:, 22:], mu_r_norm], dim=1)
        extra_x_feature = torch.cat([data.x[:, 20:], mu_r_norm], dim=1)
        graph = Data(
            x=data.x[:, :self.latent_dim],
            extra_x=extra_x_feature,
            pos=data.pos,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            ss=data.ss[:data.x.shape[0], :],
            sasa=data.x[:, 20],
            mu_r_nor=data.mu_r_norm[:data.x.shape[0], :],
            original_x=data.original_x[:, :33]
        )
        return graph


if __name__ == '__main__':
    basedir = 'dataset/cath40_k10_imem_add2ndstrc/process/'

    filelist = os.listdir(basedir)
    filelist.sort()
    random.Random(4).shuffle(filelist)
    test_filelist = filelist[-500:]
    test_dataset = Cath(test_filelist,basedir)
    data = test_dataset[10]
    print(data)

    # dl = DataLoader(train_dataset, batch_size = 10, shuffle = True, pin_memory = True, num_workers = 0)
    # for data_list in dl:
    #      print(data_list)
        #  data = Batch.from_data_list(data_list) 
        #  print(data)