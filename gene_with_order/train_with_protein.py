import torch
import torch.utils.data as tud
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns


from gene_with_order.dataset_pannel_protein import Dataset
from gene_with_order.models import transformer_cls,GPTConfig
from gene_with_order.dataset_pannel_blood import Dataset as Dataset_blood
from gene_with_order.dataset_pannel_tissue import Dataset as Dataset_tissue

class Evaluator:
    def __init__(self,cls_num):
        self.cls_num = cls_num
        self.confusion_matrix = torch.zeros(cls_num,cls_num)
        self.acc = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def update(self,pred,label):
        for i in range(len(pred)):
            self.confusion_matrix[pred[i],label[i]] += 1

    def info(self):
        self.acc = self.confusion_matrix.diag().sum()/self.confusion_matrix.sum()
        self.precision = self.confusion_matrix.diag()/self.confusion_matrix.sum(dim=0)
        self.recall = self.confusion_matrix.diag()/self.confusion_matrix.sum(dim=1)
        self.f1 = 2*self.precision*self.recall/(self.precision+self.recall)
        info = f'acc:{self.acc},precision:{self.precision},recall:{self.recall},f1:{self.f1}'
        self.sensitivity = self.confusion_matrix[1,1]/self.confusion_matrix[1,:].sum()
        self.specificity = self.confusion_matrix[0,0]/self.confusion_matrix[0,:].sum()
        info += f'sensitivity:{self.sensitivity},specificity:{self.specificity}'
        return info

    def reset(self):
        self.confusion_matrix = torch.zeros(self.cls_num,self.cls_num)
        self.acc = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def draw_confusion_matrix(self):
        # sns.heatmap(self.confusion_matrix)
        plt.figure(figsize=(3,3))
        sns.heatmap(self.confusion_matrix.T,annot=True,
                    xticklabels=['cancer','health',],
                    yticklabels=['cancer','health',],
                    cmap='Blues',fmt='g',cbar=False)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        sns.set(font_scale=1.2)
        plt.tight_layout()
        # plt.show()

    def roc_auc(self,pred,label):
        # draw roc curve
        import numpy as np
        from sklearn.metrics import roc_curve, auc

        if torch.is_tensor(pred):
            pred = pred.cpu().detach().numpy()
        if torch.is_tensor(label):
            label = label.cpu().detach().numpy()

        label = np.eye(self.cls_num)[label]
        fpr, tpr, thresholds = roc_curve(label.ravel(), pred.ravel())
        roc_auc = auc(fpr, tpr)
        self.auc = roc_auc
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                    lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        return roc_auc

    def pr_curve(self,pred,label):
        # draw pr curve
        import numpy as np
        from sklearn.metrics import precision_recall_curve, auc

        if torch.is_tensor(pred):
            pred = pred.cpu().detach().numpy()
        if torch.is_tensor(label):
            label = label.cpu().detach().numpy()

        label = np.eye(self.cls_num)[label]
        precision, recall, thresholds = precision_recall_curve(label.ravel(), pred.ravel())
        pr_auc = auc(recall, precision)
        self.auc = pr_auc
        plt.figure()
        lw = 2
        plt.plot(recall, precision, color='darkorange',
                    lw=lw, label='PR curve (area = %0.3f)' % pr_auc)
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve example')
        plt.legend(loc="lower right")
        return pr_auc


class process:
    def __init__(self, config,seg_seed,lr=1e-3,batch_size=16,
                 save_path=None
                 ):
        self.train_dataset = Dataset(type='train',seg_seed=seg_seed,config=config)
        self.val_dataset = Dataset(type='val',seg_seed=seg_seed,config=config)
        self.test_dataset = Dataset(type='test',seg_seed=seg_seed,config=config)
        self.train_dataset = tud.DataLoader(self.train_dataset,batch_size=batch_size,shuffle=True)
        self.val_dataset = tud.DataLoader(self.val_dataset,batch_size=batch_size,shuffle=True)
        self.test_dataset = tud.DataLoader(self.test_dataset,batch_size=batch_size,shuffle=True)
        self.test_dataset_ex_blood = Dataset_blood(type='test',seg_seed=seg_seed,config=config)
        self.test_dataset_ex_blood = tud.DataLoader(self.test_dataset_ex_blood,batch_size=batch_size,shuffle=True)
        self.test_dataset_ex_Tissue = Dataset_tissue(type='test',seg_seed=seg_seed,config=config)
        self.test_dataset_ex_Tissue = tud.DataLoader(self.test_dataset_ex_Tissue,batch_size=batch_size,shuffle=True)

        self.model = transformer_cls(config)
        self.fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.evaluator = Evaluator(config.cls_num)
        self.final_metric = {'AUC':0.0,'sensitivity':0.0,'specificity':0.0,'accuracy':0.0,'f1':0.0}

        if save_path is None:
            time_mark = time.strftime('%Y-%m-%d_%H_%M', time.localtime())
            os.makedirs(f'./log/{time_mark}', exist_ok=True)
            save_path = f'./log/{time_mark}'
        else:
            os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.log = ['param:',config.__dict__]

    def train_one_step(self):
        self.model.train()
        loss_sum = []
        for gene,ehr,label,pr in self.train_dataset:
            gene = gene.to(self.model.device)
            ehr = ehr.to(self.model.device)
            label = label.to(self.model.device)
            pr = pr.to(self.model.device)
            # print(gene.shape,ehr.shape,label.shape,pr.shape)
            self.optimizer.zero_grad()
            output = self.model(gene,ehr,pr)
            loss = self.fn(output,label)
            loss.backward()
            self.optimizer.step()
            loss_sum.append(loss.item())
            self.evaluator.update(output.argmax(dim=1),label)
        self.log.append({'train_loss':sum(loss_sum)/len(loss_sum)})
        self.log.append(self.evaluator.info())
        accuracy = self.evaluator.acc
        self.log.append('-'*50)
        self.evaluator.reset()
        return sum(loss_sum)/len(loss_sum),accuracy

    def val_one_step(self):
        self.model.eval()
        loss_sum = []
        for gene,ehr,label,pr in self.val_dataset:
            gene = gene.to(self.model.device)
            ehr = ehr.to(self.model.device)
            label = label.to(self.model.device)
            pr = pr.to(self.model.device)
            # print(gene.shape,ehr.shape,label.shape)
            output = self.model(gene,ehr,pr)
            loss = self.fn(output,label)
            loss_sum.append(loss.item())
            self.evaluator.update(output.argmax(dim=1),label)
        self.log.append({'val_loss':sum(loss_sum)/len(loss_sum)})
        self.log.append(self.evaluator.info())
        accuracy = self.evaluator.acc
        self.log.append('-'*50)
        self.evaluator.reset()
        return sum(loss_sum)/len(loss_sum),accuracy

    def test_one_step(self,draw_roc=False,draw_confusion_matrix=False):
        self.model.eval()
        loss_sum = []
        output_list = []
        label_list = []
        for gene,ehr,label,pr in self.test_dataset:
            gene = gene.to(self.model.device)
            ehr = ehr.to(self.model.device)
            label = label.to(self.model.device)
            pr = pr.to(self.model.device)

            output = self.model(gene,ehr,pr)
            loss = self.fn(output,label)
            loss_sum.append(loss.item())
            output_list.append(output)
            label_list.append(label)
            self.evaluator.update(output.argmax(dim=1), label)

        self.log.append({'test_loss':sum(loss_sum)/len(loss_sum)})
        self.log.append(self.evaluator.info())
        self.log.append('-'*50)
        # self.evaluator.draw_confusion_matrix()
        if draw_roc:
            output = torch.cat(output_list)
            label = torch.cat(label_list)
            self.evaluator.roc_auc(output, label)
            plt.savefig(f'{self.save_path}/roc_auc.svg')
            self.evaluator.pr_curve(output,label)
            plt.savefig(f'{self.save_path}/pr_curve.svg')
            plt.close()

        if draw_confusion_matrix:
            self.evaluator.draw_confusion_matrix()
            plt.savefig(f'{self.save_path}/confusion_matrix.svg')
            plt.close()

        if draw_roc or draw_confusion_matrix:
            output = torch.cat(output_list)
            label = torch.cat(label_list)
            self.final_metric['AUC'] = self.evaluator.roc_auc(output, label)
            self.final_metric['sensitivity'] = self.evaluator.sensitivity
            self.final_metric['specificity'] = self.evaluator.specificity
            self.final_metric['accuracy'] = self.evaluator.acc
            self.final_metric['f1'] = self.evaluator.f1

        accuracy = self.evaluator.acc
        self.evaluator.reset()
        return sum(loss_sum)/len(loss_sum),accuracy

    def test_one_step_ex_Tissue(self,draw_roc=False,draw_confusion_matrix=False):
        self.model.eval()
        loss_sum = []
        output_list = []
        label_list = []
        for gene,ehr,label,pr in self.test_dataset_ex_Tissue:
            gene = gene.to(self.model.device)
            ehr = ehr.to(self.model.device)
            label = label.to(self.model.device)
            pr = pr.to(self.model.device)

            output = self.model(gene,ehr,pr)
            loss = self.fn(output,label)
            loss_sum.append(loss.item())
            output_list.append(output)
            label_list.append(label)
            self.evaluator.update(output.argmax(dim=1), label)

        self.log.append({'test_loss':sum(loss_sum)/len(loss_sum)})
        self.log.append(self.evaluator.info())
        self.log.append('-'*50)
        # self.evaluator.draw_confusion_matrix()
        if draw_roc:
            output = torch.cat(output_list)
            label = torch.cat(label_list)
            self.evaluator.roc_auc(output, label)
            plt.savefig(f'{self.save_path}/ex_Tissue_roc_auc.svg')
            self.evaluator.pr_curve(output,label)
            plt.savefig(f'{self.save_path}/ex_Tissue_pr_curve.svg')
            plt.close()

        if draw_confusion_matrix:
            self.evaluator.draw_confusion_matrix()
            plt.savefig(f'{self.save_path}/ex_Tissue_confusion_matrix.svg')
            plt.close()

        if draw_roc or draw_confusion_matrix:
            output = torch.cat(output_list)
            label = torch.cat(label_list)
            self.final_metric['AUC'] = self.evaluator.roc_auc(output, label)
            self.final_metric['sensitivity'] = self.evaluator.sensitivity
            self.final_metric['specificity'] = self.evaluator.specificity
            self.final_metric['accuracy'] = self.evaluator.acc
            self.final_metric['f1'] = self.evaluator.f1

        accuracy = self.evaluator.acc
        self.evaluator.reset()
        return sum(loss_sum)/len(loss_sum),accuracy

    def test_one_step_ex_blood(self,draw_roc=False,draw_confusion_matrix=False):
        self.model.eval()
        loss_sum = []
        output_list = []
        label_list = []
        for gene,ehr,label,pr in self.test_dataset_ex_blood:
            gene = gene.to(self.model.device)
            ehr = ehr.to(self.model.device)
            label = label.to(self.model.device)
            pr = pr.to(self.model.device)

            output = self.model(gene,ehr,pr)
            loss = self.fn(output,label)
            loss_sum.append(loss.item())
            output_list.append(output)
            label_list.append(label)
            self.evaluator.update(output.argmax(dim=1), label)

        self.log.append({'test_loss':sum(loss_sum)/len(loss_sum)})
        self.log.append(self.evaluator.info())
        self.log.append('-'*50)
        # self.evaluator.draw_confusion_matrix()
        if draw_roc:
            output = torch.cat(output_list)
            label = torch.cat(label_list)
            self.evaluator.roc_auc(output, label)
            plt.savefig(f'{self.save_path}/ex_blood_roc_auc.svg')
            self.evaluator.pr_curve(output,label)
            plt.savefig(f'{self.save_path}/ex_blood_pr_curve.svg')
            plt.close()

        if draw_confusion_matrix:
            self.evaluator.draw_confusion_matrix()
            plt.savefig(f'{self.save_path}/ex_blood_confusion_matrix.svg')
            plt.close()

        if draw_roc or draw_confusion_matrix:
            output = torch.cat(output_list)
            label = torch.cat(label_list)
            self.final_metric['AUC'] = self.evaluator.roc_auc(output, label)
            self.final_metric['sensitivity'] = self.evaluator.sensitivity
            self.final_metric['specificity'] = self.evaluator.specificity
            self.final_metric['accuracy'] = self.evaluator.acc
            self.final_metric['f1'] = self.evaluator.f1

        accuracy = self.evaluator.acc
        self.evaluator.reset()
        return sum(loss_sum)/len(loss_sum),accuracy

    def train(self,epoch):
        best_acc = 0
        best_model = None
        for i in range(epoch):
            self.log.append(f'epoch:{i}')
            self.log.append('train')
            self.train_one_step()
            self.log.append('val')
            oss,val_acc = self.val_one_step()
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = self.model.state_dict()
            self.log.append('test')
            self.test_one_step()
            self.log.append('test_ex_blood')
        self.test_one_step_ex_blood(draw_roc=True,draw_confusion_matrix=True)
        self.test_one_step_ex_Tissue(draw_roc=True,draw_confusion_matrix=True)
        self.model.load_state_dict(best_model)

        return self.log

    def save_log(self):
        log = '\n'.join([str(i) for i in self.log])
        with open(f'{self.save_path}/log.txt','w') as f:
            f.write(log)
    def save_model(self):
        torch.save(self.model.state_dict(),f'{self.save_path}/model.pth')

    def load_model(self):
        self.model.load_state_dict(torch.load(f'{self.save_path}/model.pth'))

    def save_confusion_matrix(self):
        self.test_one_step(draw_confusion_matrix=True)

    def save_roc_auc(self):
        self.test_one_step(draw_roc=True)

    def save_everything(self):

        self.save_model()
        self.save_confusion_matrix()
        self.save_roc_auc()
        self.save_log()
        return self.final_metric


if __name__ == '__main__':
    layer = 4
    seed = 11
    config = dict(block_size=15, vocab_size=250, n_layer=layer, n_head=16, n_embd=64, dropout=0.15, bias=True, cls_num=2,
                    apply_ehr = False,if_vaf_sort=True,if_qpcr=True,if_protein=True)
    path = 'RFlog_revised_with_protein/layer/'+str(layer)+'/'+str(seed)
    config = GPTConfig(**config)
    p = process(config,seg_seed=seed,lr=1e-3,batch_size=256,save_path=path)
    p.train(20)
    print(p.log)
    p.save_everything()


    layer = 4
    seed = 11

    config = dict(block_size=15, vocab_size=250, n_layer=layer, n_head=16, n_embd=64, dropout=0.15, bias=True, cls_num=2,
                    apply_ehr = False,if_vaf_sort=True,if_qpcr=True,if_protein=False)
    path = 'RFlog_revised_without_protein/layer/'+str(layer)+'/'+str(seed)
    config = GPTConfig(**config)
    p = process(config,seg_seed=seed,lr=1e-3,batch_size=256,save_path=path)
    # p.load_model()
    p.train(20)
    print(p.log)
    # p.test_one_step_ex_blood(draw_roc=True,draw_confusion_matrix=True)
    p.save_everything()

