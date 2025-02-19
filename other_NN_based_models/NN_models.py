import torch

from einops import rearrange
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from ..train.Lora import *
from ..train.Multi_head_attention import *
from compared_nets import *
from Transformer import *


class Model(nn.Module):
    def __init__(self, gpt_type='gpt2', gpt_layers=6, heigth=100, width=50, in_len=8, out_len=8, d_model=768):
        super(Model, self).__init__()
        self.heigth = heigth  # 100
        self.width = width  # 50
        self.in_len = in_len  # 16
        self.out_len = out_len  # 16
        self.d_model = d_model  # 768
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ##
        self.linear11 = nn.Linear(in_features=self.in_len * self.heigth * self.width,
                                  out_features=self.in_len * self.d_model)
        self.cross_atten = MultiHeadAttention(self.d_model)

        self.linear1 = nn.Linear(in_features=self.in_len * 2 * self.d_model, out_features=self.out_len * self.d_model)
        self.linear2 = nn.Linear(in_features=self.out_len * self.d_model,
                                 out_features=self.out_len * 2 * self.heigth * self.width)

        #######################LSTM#######################
        #self.LSTM = LSTM(features=self.d_model, input_size=self.d_model, hidden_size=self.d_model, num_layers=6)
        # #######################RNN##############
        #self.RNN = RNN(features=self.d_model, input_size=self.d_model, hidden_size=self.d_model, num_layers=6)
        # #######################GRU########################
        #self.GRU = GRU(features=self.d_model, input_size=self.d_model, hidden_size=self.d_model, num_layers=6)
        #########################Transformer#############
        self.Transformer = Block(dim=self.d_model)
        #########################################################

    def forward(self, x):
        # x: (B,T,2,100,50)
        x_real = x[:, :, 0, :, :]  # B,T,100,50
        x_imag = x[:, :, 1, :, :]

        x_real = rearrange(x_real, 'b c h w -> b (c h w)')  # (B,T,100,50)->(B,T*100*50)
        x_imag = rearrange(x_imag, 'b c h w -> b (c h w)')  # (B,T,100,50)->(B,T*100*50)

        # (B,T*100*50)->(B,T*768)->(B,T,768)
        x_real = rearrange(self.linear11(x_real), 'b (h w) -> b h w', h=self.in_len, w=self.d_model)
        x_imag = rearrange(self.linear11(x_imag), 'b (h w) -> b h w', h=self.in_len, w=self.d_model)

        # cross-attention: (B,T,768)->(B,T,768)
        x_real_attn = self.cross_atten(x_real, x_real, x_real)
        x_imag_attn = self.cross_atten(x_imag, x_imag, x_imag)

        x = torch.stack((x_real_attn, x_imag_attn), dim=2)  # (B,T,768)->(B,T,2,768)
        ############
        x = rearrange(x, 'a b c d -> a (b c d)')  # (B,T,2,768)->(B,T*2*768)
        x = self.linear1(x).reshape(-1, self.out_len, self.d_model)  # (B,T*2*768)->(B,F*768)->(B,F,768)

        ########################################################
        y_gpt2 = self.Transformer(x)  # (B,F,768)->(B,F,768)
        #y_gpt2 = self.LSTM(x, self.out_len, self.device)  # (B,F,768)->(B,F,768)
        #y_gpt2 = self.GRU(x, self.out_len, self.device)  # (B,F,768)->(B,F,768)
        #y_gpt2 = self.RNN(x, self.out_len, self.device)  # (B,F,768)->(B,F,768)
        #################################

        y_gpt2 = rearrange(y_gpt2, 'a b c -> a (b c)')  # (B,F,768)->(B,F*768)
        y = self.linear2(y_gpt2)  # (B,F*768)->(B,F*2*100*50)
        y = rearrange(y, 'a (b c d e)->a b c d e', b=self.out_len, c=2, d=self.heigth,
                      e=self.width)  # (B,F*2*100*50)->(B,F,2,100,50)

        return y