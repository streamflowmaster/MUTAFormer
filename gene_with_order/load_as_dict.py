import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_as_dict(path = 'RFdata_hg38_VAF_Revised.xlsx',if_vaf_sort=True):
    """
    Load a model from a file and return it as a dictionary.

     Sample_ID   Class         mut_pos  ... Age    AFP PIVKAII
0     CLCA_0001     HCC   chr1:46363482  ...  63  2.900     0.5
1     CLCA_0001     HCC  chr1:113198368  ...  63  2.900     0.5
2     CLCA_0001     HCC   chr10:3909656  ...  63  2.900     0.5
3     CLCA_0001     HCC  chr10:89003179  ...  63  2.900     0.5
4     CLCA_0001     HCC  chr11:88597231  ...  63  2.900     0.5
    """
    data = pd.read_excel(path)
    gender_dict = {'Male':0,'Female':1}
    gene_dict = {}
    ehr_dict = {}
    label_dict = {}
    vaf_dict = {}
    for i in range(len(data)):
        if data['Sample_ID'][i] not in gene_dict:
            gene_dict [data['Sample_ID'][i]] = [] if pd.isnull(data['mut_pos'][i]) else [data['mut_pos'][i]]
            ehr_dict[data['Sample_ID'][i]] = [data['Age'][i], data['AFP'][i], gender_dict[data['Gender'][i]]]
            label_dict[data['Sample_ID'][i]] = data['Class'][i]
            vaf_dict[data['Sample_ID'][i]] = [] if pd.isnull(data['VAF'][i]) else [data['VAF'][i]]
        else:
            gene_dict[data['Sample_ID'][i]].append(data['mut_pos'][i])
            vaf_dict[data['Sample_ID'][i]].append(data['VAF'][i])

    if if_vaf_sort:
        for key in vaf_dict:
            idx = np.argsort(vaf_dict[key])[::-1]
            gene_dict[key] = [gene_dict[key][i] for i in idx]
            vaf_dict[key] = [vaf_dict[key][i] for i in idx]
    # print(gene_dict)
    # print(ehr_dict)
    # print(label_dict)

    return gene_dict,ehr_dict,label_dict,vaf_dict

def load_as_dict_with_protein(path = 'RFdata_hg38_VAF_Revised.xlsx',if_vaf_sort=True, if_protein=False):
    """
    Load a model from a file and return it as a dictionary.

     Sample_ID   Class         mut_pos  ... Age    AFP PIVKAII
0     CLCA_0001     HCC   chr1:46363482  ...  63  2.900     0.5
1     CLCA_0001     HCC  chr1:113198368  ...  63  2.900     0.5
2     CLCA_0001     HCC   chr10:3909656  ...  63  2.900     0.5
3     CLCA_0001     HCC  chr10:89003179  ...  63  2.900     0.5
4     CLCA_0001     HCC  chr11:88597231  ...  63  2.900     0.5
    """
    data = pd.read_excel(path)
    gender_dict = {'Male':0,'Female':1}
    gene_dict = {}
    ehr_dict = {}
    label_dict = {}
    vaf_dict = {}
    AFP_protein_dict = {}
    PIVKAII_protein_dict = {}
    for i in range(len(data)):
        if data['Sample_ID'][i] not in gene_dict:
            gene_dict [data['Sample_ID'][i]] = [] if pd.isnull(data['mut_pos'][i]) else [data['mut_pos'][i]]
            ehr_dict[data['Sample_ID'][i]] = [data['Age'][i], data['AFP'][i], gender_dict[data['Gender'][i]]]
            label_dict[data['Sample_ID'][i]] = data['Class'][i]
            vaf_dict[data['Sample_ID'][i]] = [] if pd.isnull(data['VAF'][i]) else [data['VAF'][i]]
            AFP_protein_dict[data['Sample_ID'][i]] = 0 if pd.isnull(data['AFP'][i]) else data['AFP'][i]
            PIVKAII_protein_dict[data['Sample_ID'][i]] = 0 if pd.isnull(data['PIVKAII'][i]) else data['PIVKAII'][i]
        else:
            gene_dict[data['Sample_ID'][i]].append(data['mut_pos'][i])
            vaf_dict[data['Sample_ID'][i]].append(data['VAF'][i])

    if if_vaf_sort:
        for key in vaf_dict:
            idx = np.argsort(vaf_dict[key])[::-1]
            gene_dict[key] = [gene_dict[key][i] for i in idx]
            vaf_dict[key] = [vaf_dict[key][i] for i in idx]

    # print(gene_dict.keys())
    # print(ehr_dict)
    # print(len(label_dict))
    if if_protein:
        return gene_dict,ehr_dict,label_dict,vaf_dict,AFP_protein_dict,PIVKAII_protein_dict
    else:
        return gene_dict,ehr_dict,label_dict,vaf_dict,None,None

def num_of_genes(gene_dict):
    """
    Return the number of genes in the dictionary.
    """
    len_dict = {}
    for key in gene_dict:
        len_dict[key] = len(gene_dict[key])

    # print(max(len_dict.values()), min(len_dict.values()))
    plt.hist(len_dict.values(), bins=5)
    plt.xlabel('Number of genes-mutation')
    plt.ylabel('Number of patients')
    plt.title('non-HCC patients mutation distribution')
    plt.xticks(range(0, 83, 5))
    plt.show()
    # print(len_dict)

def div_health_dict(label_dict,gene_dict,ehr_dict,vaf_dict):
    """
    Return a dictionary of health status of the patients.
    """
    health_gene = {}
    cancer_gene = {}
    health_ehr = {}
    cancer_ehr = {}
    health_key = 'nonHCC'
    cancer_key = 'HCC'
    health_vaf = {}
    cancer_vaf = {}
    for key in gene_dict:
        if label_dict[key] == cancer_key:
            health_gene[key] = gene_dict[key]
            health_ehr[key] = ehr_dict[key]
            health_vaf[key] = vaf_dict[key]

        else:
            cancer_gene[key] = gene_dict[key]
            cancer_ehr[key] = ehr_dict[key]
            cancer_vaf[key] = vaf_dict[key]

    return list(health_gene.values()),list(cancer_gene.values()),\
           list(health_ehr.values()),list(cancer_ehr.values()),\
           list(health_vaf.values()),list(cancer_vaf.values())

def div_health_dict_with_protein(label_dict,gene_dict,ehr_dict,vaf_dict,AFP_protein_dict=None,PIVKAII_protein_dict=None):
    """
    Return a dictionary of health status of the patients.
    """
    health_gene = {}
    cancer_gene = {}
    health_ehr = {}
    cancer_ehr = {}
    health_key = 'nonHCC'
    cancer_key = 'HCC'
    health_vaf = {}
    cancer_vaf = {}
    health_AFP_protein = {}
    cancer_AFP_protein = {}
    health_PIVKAII_protein = {}
    cancer_PIVKAII_protein = {}

    for key in gene_dict:
        if label_dict[key] == cancer_key:
            health_gene[key] = gene_dict[key]
            health_ehr[key] = ehr_dict[key]
            health_vaf[key] = vaf_dict[key]

        else:
            cancer_gene[key] = gene_dict[key]
            cancer_ehr[key] = ehr_dict[key]
            cancer_vaf[key] = vaf_dict[key]

    if AFP_protein_dict is not None:
        for key in gene_dict:
            if label_dict[key] == cancer_key:
                health_AFP_protein[key] = AFP_protein_dict[key]
            else:
                cancer_AFP_protein[key] = AFP_protein_dict[key]

    if PIVKAII_protein_dict is not None:
        for key in gene_dict:
            if label_dict[key] == cancer_key:
                health_PIVKAII_protein[key] = PIVKAII_protein_dict[key]
            else:
                cancer_PIVKAII_protein[key] = PIVKAII_protein_dict[key]

    to_return = list(health_gene.values()),list(cancer_gene.values()),\
           list(health_ehr.values()),list(cancer_ehr.values()),\
           list(health_vaf.values()),list(cancer_vaf.values())
    if AFP_protein_dict is not None:
        to_return += list(health_AFP_protein.values()),list(cancer_AFP_protein.values())
    else:
        to_return += None,None
    if PIVKAII_protein_dict is not None:
        to_return += list(health_PIVKAII_protein.values()),list(cancer_PIVKAII_protein.values())
    else:
        to_return += None,None
    return to_return

def mutation_gene_freq(gene_dict,if_save=False):
    """
    Return the frequency of each gene in the dictionary.
    """

    gene_freq = {}
    for key in gene_dict:
        for gene in gene_dict[key]:
            if gene not in gene_freq:
                gene_freq[gene] = 1
            else:
                gene_freq[gene] += 1

    vis_num = 50
    sorted_freq = np.sort(list(gene_freq.values()))[::-1]
    sorted_gene = sorted(gene_freq, key=gene_freq.get, reverse=True)
    if if_save:
        plt.figure(figsize=(5, 10), dpi=600)
        plt.barh(np.flipud(sorted_gene[:vis_num]), np.flipud(sorted_freq)[:vis_num])
        plt.yticks(rotation=0)
        # plt.ylim(200, 250)
        plt.xlim(0, 150)
        plt.subplots_adjust(left=0.3, right=0.95, top=0.99, bottom=0.01)
        path = 'gene_mutation_frequency.svg'
        plt.savefig(path, dpi=600, bbox_inches='tight')
        plt.close()
    return sorted_freq,sorted_gene


def mutation_mapping(sorted_gene,vis_num=250):
    """
    Return the mapping of the genes.
    """
    gene_map = {}
    assert vis_num <= len(sorted_gene)
    for i in range(0,vis_num):
        gene_map[sorted_gene[i]] = i+1
    return gene_map

def get_gene_map(vis_num=45):
    gene_dict, ehr_dict, label_dict,vaf_dict = load_as_dict()
    freq, gene = mutation_gene_freq(gene_dict)
    gene_map = mutation_mapping(gene,vis_num=vis_num)
    return gene_map

if __name__ == '__main__':
    gene_dict,ehr_dict,label_dict,vaf_dict = load_as_dict()
    # print(vaf_dict)
    # print(div_health_dict(label_dict,gene_dict,ehr_dict,vaf_dict))
    # print(num_of_genes(health_dict))
    # print(num_of_genes(cancer_dict))

    freq, gene = mutation_gene_freq(gene_dict)
    gene_map = mutation_mapping(gene)
    print(gene_map.keys(),
          gene_map.values())
    print(freq)
    np.savetxt('gene_freq.txt',freq.astype(int),fmt='%d')
