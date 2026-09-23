import numpy as np
import os
from captum.attr import (
    # Deconvolution as FeaturePermutation,
    # IntegratedGradients as FeaturePermutation,
    LayerGradCam,
    # GuidedGradCam as FeaturePermutation,
    # PerturbationAttribution as FeaturePermutation,
    FeaturePermutation
)
from load_as_dict import get_gene_map
from gene_with_order.models import transformer_cls,GPTConfig
from gene_with_order.models_for_explain import covered_model
from gene_with_order.dataset_qPCR import Dataset
import torch
import torch.utils.data as tud
import seaborn as sns
import matplotlib.pyplot as plt

class Explain():
    def __init__(self,config:GPTConfig,device,seg_seed:int,model_path:str='model.pth'):
        self.model = covered_model(config)
        self.model.model.load_checkpoint(model_path,device=device)
        self.dataset = Dataset(type='test',seg_seed=seg_seed,config=config)
        self.dataloader = tud.DataLoader(self.dataset,batch_size=4,shuffle=True)
        self.model.eval()
        self.features_in_hook = []
        self.features_out_hook = []
        self.config = config
        # self.gene  = list(get_gene_map(vis_num=config.vocab_size).keys())
        self.gene = list(get_gene_map(vis_num=config.vocab_size).keys())
        # self.gene = ['chr5:1295113', 'chr17:7674216', 'chr3:41224607', 'chr3:41224633', 'chr3:41224645',
        # 'chr3:41224622', 'chr1:64844820', 'chr5:1295135', 'chr3:41224609', 'chr1:64845520', 'chr3:41224610',
        # 'chr22:30424432', 'chr3:41224606', 'chr4:73417669', 'chr17:7675143', 'chr17:7675088', 'chr4:73410355',
        # 'chr17:42322464', 'chr1:67430342', 'chr3:41224621', 'chr17:7674221', 'chr17:7674953', 'chr17:7676040',
        #  'chr6:154154566', 'chr17:7675076', 'chr8:92092849', 'chr20:24995016', 'chr3:179234297', 'chr6:26103895',
        #  'chr3:170986096', 'chr1:67430357', 'chr17:7674241', 'chr3:138653763', 'chr1:214027680', 'chr3:41224646',
        #  'chr17:7673839', 'chr5:55964157', 'chr4:73409507', 'chrX:80444259', 'chr17:7675137', 'chr16:73539137',
        #  'chr2:5987461', 'chr2:53665425', 'chr9:101418516', 'chr3:58318738', 'chr14:88562935', 'chr9:35658045',
        #  'chr17:7674945', 'chr17:7676055', 'chr1:114716756', 'chr11:17382487', 'chr4:181488368', 'chr20:21715364',
        #  'chr17:7674230', 'chr14:39181988', 'chr14:103335845', 'chr20:21511447', 'chr2:111884140', 'chr3:50220304',
        #  'chr1:184805749', 'chr12:124360396', 'chr2:47807217', 'chr19:3985486', 'chr4:73409334', 'chr17:7673802',
        #  'chr8:4997182', 'chr4:154583444', 'chr9:21971112', 'chr17:8152475', 'chr17:5587617', 'chr4:154583430',
        #  'chr5:1295046', 'chr12:70628734', 'chr1:204516449', 'chr10:43671837', 'chr16:346233', 'chr4:99306006',
        #  'chr3:138653772', 'chr17:7674218', 'chr19:6720674', 'chr5:55964233', 'chr17:7670699', 'chrX:124376770',
        #  'chr2:240886243', 'chr4:73408804', 'chr16:2497578', 'chr11:126441012', 'chr19:3452575', 'chr3:41224613',
        #  'chr8:74853594', 'chr4:154571012', 'chr4:44020378', 'chr12:12916406', 'chr17:7675157', 'chr17:7674194',
        #  'chr1:46363482', 'chr1:113198368', 'chr10:3909656', 'chr10:89003179', 'chr11:88597231', 'chr13:77895780',
        #  'chr15:55619397', 'chr17:74222570', 'chr19:28461306', 'chr2:106802902', 'chr21:29596625', 'chr21:32793730',
        #  'chr21:36576549', 'chr3:136956105', 'chr3:190651173', 'chr8:22665273', 'chr10:20781757', 'chr2:200585967',
        #  'chr20:64213610', 'chr3:128935129', 'chr4:161385500', 'chr7:31529845', 'chr12:52836044', 'chr22:42955476',
        #  'chr4:42089423', 'chr9:2153829', 'chr10:15514655', 'chr10:120503782', 'chr14:100323172', 'chr19:50873533',
        #  'chr3:86072332', 'chr7:155004582', 'chr1:4782683', 'chr11:35616362', 'chr7:123948385', 'chr8:96261760',
        #  'chr1:23132106', 'chr11:61754744', 'chr2:140136518', 'chr5:140788554', 'chr6:63323199', 'chr1:31665694',
        #  'chr10:102455855', 'chr11:65427320', 'chr11:65431252', 'chr12:69818191', 'chr12:94567911', 'chr18:35491469',
        #  'chr2:209820540', 'chr2:227363286', 'chr20:61939057', 'chr8:28751352', 'chr9:106870752', 'chr1:39483169',
        #  'chr1:50978508', 'chr1:61077447', 'chr1:71404686', 'chr1:156724231', 'chr1:179592538', 'chr1:228409077',
        #  'chr1:248859006', 'chr10:37315710', 'chr10:127998828', 'chr10:132379955', 'chr11:821893', 'chr11:25081419',
        #  'chr11:62841855', 'chr11:63760963', 'chr11:70487267', 'chr11:131533863', 'chr12:15348461', 'chr12:62573561',
        #  'chr12:71610121', 'chr12:127324187', 'chr13:77007548', 'chr13:102402190', 'chr13:114291736', 'chr14:32943175',
        #  'chr14:68809148', 'chr14:81274122', 'chr14:94323284', 'chr15:42499771', 'chr15:60007370',
        #  'chr15:67526939', 'chr16:11759951', 'chr17:7673704', 'chr17:47850994', 'chr18:13240937',
        #  'chr18:24134846', 'chr19:17751458', 'chr19:21746825', 'chr19:42334146', 'chr19:48303719',
        #  'chr19:53875323', 'chr2:140950296', 'chr2:165748268', 'chr2:166405287', 'chr2:169705659',
        #  'chr2:200909353', 'chr2:206124091', 'chr2:207132168', 'chr2:239053577', 'chr20:481511',
        #  'chr20:35603379', 'chr22:32474950', 'chr22:37775788', 'chr3:9804621', 'chr3:26624008',
        #  'chr3:71586867', 'chr3:114619498', 'chr3:153164498', 'chr3:180707998', 'chr3:197884787',
        #  'chr4:36066798', 'chr4:62072322', 'chr4:78419235', 'chr4:82864412', 'chr5:15615837',
        #  'chr6:35316257', 'chr6:50844463', 'chr6:97707576', 'chr6:143061600', 'chr6:159692318',
        #  'chr7:100733928', 'chr7:107760798', 'chr8:379445', 'chr8:129786334', 'chr8:133017587',
        #  'chr8:142782670', 'chr9:95806417', 'chrX:25004261', 'chr2:211380889', 'chr11:32453592',
        #  'chr14:39177976', 'chr1:6072787', 'chr1:51060027', 'chr1:64839716', 'chr1:76632251',
        #  'chr15:96414505', 'chr3:41224634', 'chr1:9347648', 'chr1:38135827', 'chr12:128813176',
        #  'chr13:25882312', 'chr14:27639068', 'chr2:177538599', 'chr2:214577479', 'chr5:62133554',
        #  'chr8:81440514', 'chr1:155860182', 'chr16:68013856', 'chr3:16601451', 'chr14:35806765', 'chr16:71845900']


    def hook(self,module, fea_in, fea_out):
        self.features_in_hook.append(fea_in)  # 去掉这行就不会留下输入了
        self.features_out_hook.append(fea_out)
        return None

    def reset_hook(self):
        self.features_in_hook = []
        self.features_out_hook = []

    def view_hook(self,gene):
        # print(self.features_in_hook)
        # print(self.features_out_hook)
        # print(self.features_in_hook[0],
        #       self.features_out_hook[0])
        qkv,_ = self.features_out_hook[0]
        atten = torch.matmul(qkv, qkv.transpose(-1, -2))
        # print(atten[1].shape)
        for i in range(atten.shape[0]):
            plt.figure(figsize=(5,5))
            sns.heatmap(atten[i].detach().numpy(),cbar=False)
            plt.xticks(range(15)+np.array(0.5),gene[i].detach().numpy())
            plt.yticks(range(15)+np.array(0.5),gene[i].detach().numpy())
            plt.title('sample'+str(i))
            plt.show()

    def view_attention(self):
        self.model.transformer_encoder.layers[2].self_attn.register_forward_hook(self.hook)
        for gene,ehr,label in self.dataloader:
            gene = gene.to(self.model.device)
            ehr = ehr.to(self.model.device)
            output = self.model(gene,ehr)
            seq = self.model.embedding(gene)
            sns.heatmap(seq[1].detach().numpy())
            plt.show()
            break
        self.view_hook(gene)


    def explain(self,path=None):
        self.model.to('cpu')
        deconv = FeaturePermutation(self.model)
        attributions = torch.zeros((self.config.vocab_size))
        freq = torch.zeros((self.config.vocab_size))
        for gene,ehr,label in self.dataloader:
            gene = gene.to(self.model.device)
            ehr = ehr.to(self.model.device)
            label = label.to(self.model.device)
            data = gene
            # data = torch.cat([gene,ehr],dim=1)
            attribution = deconv.attribute(data,target=label).view(-1)
            gene_flatten = gene.view(-1)
            attributions[gene_flatten] += attribution
            freq[gene_flatten] += 1

        # normed_attract = attributions/freq
        normed_attract = attributions
        normed_attract = np.nan_to_num(normed_attract)
        self.draw_attribution(normed_attract,path)
        torch.save(normed_attract,os.path.join(path,'attribution.pt'))
        torch.save(attributions,os.path.join(path,'attribution_unormed.pt'))

    def draw_attribution(self,attr,path=None):
        plt.figure(figsize=(5, 10), dpi=600)
        vaccum = attr[0]
        attr = np.abs(attr[1:])
        attr_sort = np.argsort(attr)
        print(self.gene)
        gene = [self.gene[i] for i in attr_sort]
        attr = [attr[i] for i in attr_sort]
        plt.barh(gene, attr)
        plt.yticks(rotation=0)
        plt.ylim(200, 250)
        plt.subplots_adjust(left=0.3, right=0.95, top=0.99, bottom=0.01)
        if path is not None:
            # write the gene list and corresponding attribution to a file of gene_list.xlsx
            import pandas as pd
            df = pd.DataFrame({'gene':gene,'attribution':attr})
            df.to_excel(os.path.join(path,'gene_list.xlsx'))
        if path is not None:
            path = os.path.join(path, 'attribution.svg')
            plt.savefig(path, dpi=600, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    config = dict(block_size=15, vocab_size=250, n_layer=10, n_head=16, n_embd=64, dropout=0.1, bias=True, cls_num=2,
                    apply_ehr = False,)
    config = GPTConfig(**config)
    for i in range(16):
        explain = Explain(config,device='cpu',seg_seed=i,model_path='log_revised_abalation/layer/10/'+str(i)+'/model.pth')
        explain.explain(path='log_revised_abalation/layer/10/'+str(i))
    # explain = Explain(config,device='cpu',seg_seed=1,model_path='log_revised_abalation_1/layer/10/1/model.pth')
    # explain.explain(path='log_revised_abalation_1/layer/10/1')