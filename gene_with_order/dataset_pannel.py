import torch.nn
import torch.utils.data as tud
from gene_with_order.load_as_dict import load_as_dict,div_health_dict, get_gene_map
from gene_with_order.models import GPTConfig
import pandas as pd

class Dataset(tud.Dataset):
    def __init__(self, type='train',
                 seg_seed = None,
                 include_plex_num = 100,
                 seg_ratio:list=[0.7,0.1,0.2],
                 config:GPTConfig = GPTConfig(),
                 plex_name = 'plex_mapping.xlsx'):

        gene_dict,ehr_dict,label_dict,vaf_dict = load_as_dict(if_vaf_sort=config.if_vaf_sort)
        '''
        chr17:7674216-7674229
        chr17:7674948-7674964
        chr17:7675135-7675144
        chr3:41224605-41224623
        chr3:41224633-41224650
        chr5:1295104-1295116
        chr5:1295130-1295143
        chr4:73417659-73417678
        '''
        map = pd.read_excel(plex_name)
        self.cluster_map = {}
        for i in reversed(range(len(map))):
            cluster_id = map['cluster_id'][i]
            if cluster_id <= include_plex_num:
                self.cluster_map[cluster_id] = map['mut_position'][i]

        self.plex_cluster_map = {}
        for i in reversed(range(len(map))):
            cluster_id = map['cluster_id'][i]
            if cluster_id <= include_plex_num:
                mut_position = map['mut_position'][i]
                mapped_pos = self.cluster_map[cluster_id]
                self.plex_cluster_map[mut_position] = mapped_pos

        # print(self.plex_cluster_map)
        # self.PCR_bank = {
        #         0: ['chr17',7674216,7674229],
        #         1: ['chr17',7674948,7674964],
        #         2: ['chr17',7675135,7675144],
        #         3: ['chr3',41224605,41224623],
        #         4: ['chr3',41224633,41224650],
        #         5: ['chr5',1295104,1295116],
        #         6: ['chr5',1295130,1295143],
        #         7: ['chr4',73417659,73417678]
        # }
        #
        # self.PCR_chrom = {'chr17':[0,1,2],'chr3':[3,4],'chr5':[5,6],'chr4':[7]}
        #
        # self.pcr_dict = {
        #     0: 'chr17:7674216',
        #     1: 'chr17:7674953',
        #     2: 'chr17:7675143',
        #     3: 'chr3:41224607',
        #     4: 'chr3:41224633',
        #     5: 'chr5:1295113',
        #     6: 'chr5:1295135',
        #     7: 'chr4:73417669'
        #
        # }

        self.health_gene,self.cancer_gene,self.health_ehr,self.cancer_ehr,_,_ = div_health_dict(label_dict,gene_dict,ehr_dict,vaf_dict)
        self.type = type
        self.seg_ratio = seg_ratio
        self.selected_gene,self.selected_ehr = self.dataseg(seg_seed)
        self.vocab_size = config.vocab_size
        self.map = get_gene_map(vis_num=config.vocab_size-2)
        self.max_gene_num = config.block_size
        self.if_qpcr = config.if_qpcr

    def dataseg(self,seg_seed):
        if seg_seed is not None:
            torch.manual_seed(seg_seed)
            h_idxs = torch.randperm(
                len(self.health_gene))
            c_idxs = torch.randperm(
                len(self.cancer_gene))

            h_train_size = int(len(self.health_gene) * self.seg_ratio[0])
            h_val_size = int(len(self.health_gene) * self.seg_ratio[1])
            h_test_size = len(self.health_gene) - h_train_size - h_val_size

            h_train_idx = h_idxs[:h_train_size]
            h_val_idx = h_idxs[h_train_size:h_train_size + h_val_size]
            h_test_idx = h_idxs[h_train_size + h_val_size:]
            if self.type == 'train':
                h_idxs = h_train_idx
            elif self.type == 'val':
                h_idxs = h_val_idx
            elif self.type == 'test':
                h_idxs = h_test_idx
            else:
                raise ValueError('type must be train,val or test')

            c_train_size = int(len(self.cancer_gene) * self.seg_ratio[0])
            c_val_size = int(len(self.cancer_gene) * self.seg_ratio[1])
            c_test_size = len(self.cancer_gene) - c_train_size - c_val_size
            c_train_idx = c_idxs[:c_train_size]
            c_val_idx = c_idxs[c_train_size:c_train_size + c_val_size]
            c_test_idx = c_idxs[c_train_size + c_val_size:]
            print(c_train_size,c_val_size,c_test_size,
                  h_train_size,h_val_size,h_test_size)
            if self.type == 'train':
                c_idxs = c_train_idx
            elif self.type == 'val':
                c_idxs = c_val_idx
            elif self.type == 'test':
                c_idxs = c_test_idx
            else:
                raise ValueError('type must be train,val or test')
            self.selected_gene = [self.health_gene[i] for i in h_idxs]
            self.selected_gene += [self.cancer_gene[i] for i in c_idxs]
            self.selected_ehr = [self.health_ehr[i] for i in h_idxs]
            self.selected_ehr += [self.cancer_ehr[i] for i in c_idxs]
            self.select_label = [0] * len(h_idxs) + [1] * len(c_idxs)

            if self.type == 'train':
                self.selected_gene += [self.health_gene[i] for i in h_idxs]
                self.select_label += [0] * len(h_idxs)
                self.selected_ehr += [self.health_ehr[i] for i in h_idxs]
        else:
            raise ValueError('seg_seed must be set')
        return self.selected_gene,self.selected_ehr

    def map_gene(self,gene):
        mapped_gene = []
        for g in gene:
            if g in self.map:
                mapped_gene.append(self.map[g])
            else:
                mapped_gene.append(self.vocab_size-1)

        if len(mapped_gene) > self.max_gene_num:
            mapped_gene = mapped_gene[:self.max_gene_num]
        else:
            mapped_gene += [0] * (self.max_gene_num - len(mapped_gene))
        mapped_gene = torch.tensor(mapped_gene)
        return mapped_gene

    def make_pcr_format(self,gene):
        if gene in self.plex_cluster_map:
                return self.plex_cluster_map[gene]
        return 'None'

    def batch_make_pcr_format(self,gene_list):
        pcr_list = []
        for gene in gene_list:
            pcr_list.append(self.make_pcr_format(gene))
        # 'remove None'
        pcr_list = [pcr for pcr in pcr_list if pcr != 'None']
        return pcr_list

    def __len__(self):
        # print(len(self.selected_gene))
        return len(self.selected_gene)

    def __getitem__(self, index):
        if self.if_qpcr:
            selected = self.selected_gene[index]
            selected = self.batch_make_pcr_format(selected)
            if len(selected) == 0:
                return torch.tensor([0] * self.max_gene_num),torch.tensor(self.selected_ehr[index]).float(),torch.tensor(self.select_label[index])
            return self.map_gene(selected),torch.tensor(self.selected_ehr[index]).float(),torch.tensor(self.select_label[index])

        return self.map_gene(self.selected_gene[index]),torch.tensor(self.selected_ehr[index]).float(),torch.tensor(self.select_label[index])


if __name__ == '__main__':
    dataset = Dataset(type='train',seg_seed=1)
#     print()
#     print(dataset[1])
    # print(len(dataset))
    # print(dataset.map_gene(dataset.selected_gene[0]))
    # print(dataset.selected_gene[0])
    # print(dataset.selected_ehr[0])
    # print(dataset.selected_gene[0])
    # print(dataset.selected_ehr[0])

