from dataclasses import dataclass
import torch
import torch.nn as nn
@dataclass

class GPTConfig:
    block_size: int = 64
    vocab_size: int = 150 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 4
    n_head: int = 16
    n_embd: int = 64
    dropout: float = 0.1
    bias: bool = True
    cls_num: int = 2
    apply_ehr: bool = False
    if_vaf_sort: bool = False
    if_qpcr: bool = False
    if_protein: bool = False

class transformer_cls(nn.Module):

    def __init__(self, config: GPTConfig):
        super(transformer_cls, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        encoder_layers = nn.TransformerEncoderLayer(d_model=config.n_embd, nhead=config.n_head, dim_feedforward=config.n_embd,
                                                    dropout=config.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=config.n_layer,)
        if config.if_protein and config.apply_ehr:
            self.protein_encoder = MLP(2,6)
            self.ehr_encoder = MLP(3,6)
            self.fc = nn.Linear(config.n_embd+6+6, config.cls_num)
        elif config.apply_ehr and not config.if_protein:
            self.ehr_encoder = MLP(3,6)
            self.fc = nn.Linear(config.n_embd+6, config.cls_num)

        elif config.if_protein and not config.apply_ehr:
            self.protein_encoder = MLP(2,6)
            self.fc = nn.Linear(config.n_embd+6, config.cls_num)
        else:
            self.fc = nn.Linear(config.n_embd, config.cls_num)
        self.config = config
        self.device =  'cpu'
        self.to(self.device)

    def forward(self, seq,ehr=None,protein=None):

        seq = self.embedding(seq)
        if self.config.if_vaf_sort:
            pe = self.position_encoding(seq.shape[1], seq.shape[2])
            seq = seq + pe

        output = self.transformer_encoder(seq)
        output = torch.mean(output, dim=1)
        if self.config.apply_ehr:
            ehr = self.ehr_encoder(ehr)
            output = torch.cat([output, ehr], dim=1)
        if self.config.if_protein:
            protein = self.protein_encoder(protein)
            output = torch.cat([output, protein], dim=1)
        output = self.fc(output)
        output = torch.softmax(output, dim=1)
        return output

    def load_checkpoint(self, path,device):
        self.load_state_dict(torch.load(path,map_location=device))
        self.eval()

    def position_encoding(self, seq_len, n_embd):
        pe = torch.zeros(seq_len, n_embd)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-torch.log(torch.tensor(10000.0)) / n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe



class MLP(nn.Module):
    def __init__(self, embed,out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embed, embed*3)
        self.fc2 = nn.Linear(embed*3, embed*9)
        self.fc3 = nn.Linear(embed*9, out)

    def forward(self,input):
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    pdict = dict(block_size=64, vocab_size=150, n_layer=4, n_head=16, n_embd=64, dropout=0.1, bias=True, cls_num=2,
                 if_vaf_sort=True, if_qpcr=False, apply_ehr=False)
    config = GPTConfig(**pdict)
    model = transformer_cls(config)
    seq = torch.randint(0, config.vocab_size, (1, config.block_size)).to(model.device)
    output = model(seq,0)
    print(seq.shape,output.shape)