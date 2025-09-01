import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
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
            # nn.LeakyReLU(negative_slope=0.01),
            # nn.GELU(),
            nn.SiLU(),
            nn.Linear(in_features=2048, out_features=self.out_len * 2 * self.heigth * self.width),
        )

        #######################GPT-2#######################
        # / home / edu / ZhangYali / LLM_MA / GPT2 / Base
        # GPT-2
        self.gpt2 = GPT2Model.from_pretrained('/home/edu/ZhangYali/LLM_MA/GPT2/Base', output_attentions=True,
                                              output_hidden_states=True)
        # tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('/home/edu/ZhangYali/LLM_MA/GPT2/Base')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        #
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.gpt_dim = 768
        self.gpt2 = self.gpt2.to(device=self.device)

        # freeze GPT-2
        for param in self.gpt2.parameters():
            param.requires_grad = False

        # dymanic prompt
        self.prompt_template = (
            "Fluid Antenna Moving Ports Prediction Task:\n"
            "Input: Channel tables for preceding {T} time steps\n"
            "Dataset Description: The channel table is complex, and we split the channel table into real and imaginary parts\n"
            "Statistics:\n"
            "- Minimum value: {min_val_real:.4f}, {min_val_imag:.4f}\n"
            "- Maximum value: {max_val_real:.4f}, {max_val_imag:.4f}\n"
            "- Mean value: {mean_real:.4f}, {mean_imag:.4f}\n"
            "- Standard deviation: {std_real:.4f}, {std_imag:.4f}\n"
            "- Median value: {median_real:.4f}, {median_imag:.4f}\n"
            "Output: Predict channel tables for subsequent {F} time steps"
        )

        # prompt encoder
        self.prompt_encoder = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Linear(768, 768)
        )
        self.current_prompt = None
        self.prompt_length = None

        # statistics cache
        self.last_stats = {}
        #########################################################
    def calculate_statistics(self, x_real, x_imag):
        """calculate statistics of input data"""
        x_real_flatten = x_real.flatten(start_dim=1)  
        x_imag_flatten = x_imag.flatten(start_dim=1)

        stats = {
            'min_val_real': x_real_flatten.min(dim=1)[0].min().item(),
            'min_val_imag': x_imag_flatten.min(dim=1)[0].min().item(),
            'max_val_real': x_real_flatten.max(dim=1)[0].max().item(),
            'max_val_imag': x_imag_flatten.max(dim=1)[0].max().item(),
            'mean_real': x_real_flatten.mean(dim=1).mean().item(),
            'mean_imag': x_imag_flatten.mean(dim=1).mean().item(),
            'std_real': x_real_flatten.std(dim=1).mean().item(),
            'std_imag': x_imag_flatten.std(dim=1).mean().item(),
            'median_real': x_real_flatten.median(dim=1).values.mean().item(),
            'median_imag': x_imag_flatten.median(dim=1).values.mean().item(),
        }
        self.last_stats = stats 
        return stats

    def generate_dynamic_prompt(self, x_real, x_imag):
        """generate dynamic prompt based on input data"""
        # calculate statistics
        stats = self.calculate_statistics(x_real, x_imag)

        # generate prompt
        prompt_text = self.prompt_template.format(
            T=self.in_len,
            F=self.out_len,
            **stats
        )

        # tokenize prompt
        # inputs.shape (1,max_length)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            max_length=100,
            truncation=True,
            padding='max_length'
        ).to(self.device)

        # embedding.shape (1,max_length,768)
        with torch.no_grad():
            embeddings = self.gpt2.get_input_embeddings()(inputs.input_ids)

        # encode trainable prompt
        self.current_prompt = self.prompt_encoder(embeddings) # (1,max_length,768)
        self.prompt_length = inputs.input_ids.size(1) #length of prompt

    def forward(self, x):
        # x: (B,T,2,100,50)
        x_real = x[:, :, 0, :, :]  # B,T,100,50
        x_imag = x[:, :, 1, :, :]

        # prompt
        x_real_prompt = rearrange(x_real, 'b c h w -> b (c h w)')  # (B,T*100*50)
        x_imag_prompt = rearrange(x_imag, 'b c h w -> b (c h w)')

        # generate dynamic prompt
        self.generate_dynamic_prompt(x_real_prompt, x_imag_prompt)

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

        ####################### Prompt #######################
        batch_size = x.size(0)
        prompt_embeds = self.current_prompt.expand(batch_size, -1, -1)

        # cat prompt and input
        inputs_embeds = torch.cat([prompt_embeds, x], dim=1)
        # gpt2
        gpt_outputs = self.gpt2(inputs_embeds=inputs_embeds)
        y_gpt2 = gpt_outputs.last_hidden_state[:, self.prompt_length:, :]
        ###########################################################

        y_gpt2 = rearrange(y_gpt2, 'a b c -> a (b c)')  # (B,F,768)->(B,F*768)
        y = self.linear2(y_gpt2)  # (B,F*768)->(B,F*2*100*50)
        y = rearrange(y, 'a (b c d e)->a b c d e', b=self.out_len, c=2, d=self.heigth,
                      e=self.width)  # (B,F*2*100*50)->(B,F,2,100,50)

        return y

    def get_prompt_info(self):
        """obtain prompt information"""
        return {
            'prompt_text': self.prompt_template.format(
                T=self.in_len,
                F=self.out_len,
                **self.last_stats
            ),
            'statistics': self.last_stats
        }