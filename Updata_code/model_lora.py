import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from Lora_new import *
from Multi_head_attention import *


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1, 2)

    def forward(self, x):
        return self.conv(x) + self.skip(x)


class Model(nn.Module):
    def __init__(self, gpt_type='gpt2', gpt_layers=6, heigth=100, width=50, in_len=8, out_len=8, d_model=768):
        super(Model, self).__init__()
        self.heigth = heigth  # 100
        self.width = width  # 50
        self.in_len = in_len  # 16
        self.out_len = out_len  # 16
        self.d_model = d_model  # 768
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #######(B,T,100,50)->(B,T*768)
        self.linear11 = nn.Sequential(
            # 
            DownSampling(in_channels=in_len, out_channels=in_len),  # (B,T,100,50)->(B,T,50,25)

            #
            DownSampling(in_channels=in_len, out_channels=in_len),  # (B,T,50,25)->(B,T,25,13)

            #
            nn.Flatten(),
            nn.Linear(in_len * 25 * 13, in_len * d_model),
            nn.LeakyReLU()
        )

        self.cross_atten = MultiHeadAttention(self.d_model)

        self.linear1 = nn.Sequential(
            nn.Linear(in_features=self.in_len * 2 * self.d_model, out_features=2048),
            nn.GELU(),
            nn.Linear(in_features=2048, out_features=self.out_len * self.d_model),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(in_features=self.out_len * self.d_model, out_features=2048),
            #nn.LeakyReLU(negative_slope=0.01),
            #nn.GELU(),
            nn.SiLU(),
            nn.Linear(in_features=2048, out_features=self.out_len * 2 * self.heigth * self.width),
        )

        #######################GPT-2#######################
        # / home / edu / ZhangYali / LLM_MA / GPT2 / Base
        # GPT-2
        self.gpt2 = GPT2Model.from_pretrained('/home/edu/ZhangYali/LLM_MA/GPT2/Base', output_attentions=True,
                                              output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.gpt_dim = 768

        self.gpt2 = self.gpt2.to(device=self.device)

        #
        self.Lora_gpt2 = LoRA_gpt2(model=self.gpt2, r=4, gpt_dim=self.gpt_dim)
        
        # full_tuning
        #for i, (name, param) in enumerate(self.gpt2.named_parameters()):
        #    param.requires_grad = True
            
            
        # freeze
        # for i, (name, param) in enumerate(self.gpt2.named_parameters()):
        #     if 'ln' in name or 'wpe' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        #########################################################

    def forward(self, x):
        # x: (B,T,2,100,50)
        x_real = x[:, :, 0, :, :]  # B,T,100,50
        x_imag = x[:, :, 1, :, :]

        # x_real = rearrange(x_real, 'b c h w -> b (c h w)')  # (B,T,100,50)->(B,T*100*50)
        # x_imag = rearrange(x_imag, 'b c h w -> b (c h w)')  # (B,T,100,50)->(B,T*100*50)

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

        ############################GPT-2############################
        y_gpt2 = self.Lora_gpt2(x)  # (B,F,768)->(B,F,768)
        #y_gpt2 = self.gpt2(inputs_embeds=x).last_hidden_state  # (B,F,768)->(B,F,768)
        ###############GPT-2##################

        y_gpt2 = rearrange(y_gpt2, 'a b c -> a (b c)')  # (B,F,768)->(B,F*768)
        y = self.linear2(y_gpt2)  # (B,F*768)->(B,F*2*100*50)
        y = rearrange(y, 'a (b c d e)->a b c d e', b=self.out_len, c=2, d=self.heigth,
                      e=self.width)  # (B,F*2*100*50)->(B,F,2,100,50)

        return y


if __name__ == "__main__":
    input = torch.randn((2, 8, 2, 100, 50))
    print(input.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    input = input.to(device)
    model = Model().to(device)
    output = model(input)
    print('output:', output.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))