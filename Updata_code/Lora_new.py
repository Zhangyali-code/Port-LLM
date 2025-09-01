import torch
import torch.nn as nn
import math 


# �޸ĺ��_LoRA_qkv��
class _LoRA_qkv(nn.Module):
    def __init__(self, qkv, dim, r):
        super().__init__()
        self.qkv = qkv
        self.dim = dim

        # ��LoRA����ע��Ϊģ������
        self.linear_a_q = nn.Parameter(torch.zeros(r, dim))
        self.linear_b_q = nn.Parameter(torch.zeros(dim, r))
        self.linear_a_v = nn.Parameter(torch.zeros(r, dim))
        self.linear_b_v = nn.Parameter(torch.zeros(dim, r))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.linear_a_q, a=math.sqrt(5))
        nn.init.zeros_(self.linear_b_q)  # ����B��ʼ��Ϊ0
        nn.init.kaiming_uniform_(self.linear_a_v, a=math.sqrt(5))
        nn.init.zeros_(self.linear_b_v)  # ����B��ʼ��Ϊ0

    def forward(self, x):
        qkv = self.qkv(x)
        # ֱ��ʹ��ģ������
        new_q = x @ self.linear_a_q.T @ self.linear_b_q.T
        new_v = x @ self.linear_a_v.T @ self.linear_b_v.T
        qkv[..., :self.dim] += new_q
        qkv[..., -self.dim:] += new_v
        return qkv


# �޸ĺ��LoRA_gpt2��
class LoRA_gpt2(nn.Module):
    def __init__(self, model, r: int, gpt_dim=768):
        super().__init__()

        # # ����ԭʼ��������LayerNorm��λ�ñ��룩
        # for name, param in model.named_parameters():
        #     param.requires_grad = ('ln' in name or 'wpe' in name)

        # freeze image encoder first
        for i, (name, param) in enumerate(model.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # �洢����LoRA��
        self.lora_layers = nn.ModuleList()

        for blk in model.h:
            # ����LoRA�㲢ע��
            lora_layer = _LoRA_qkv(blk.attn.c_attn, dim=gpt_dim, r=r)
            self.lora_layers.append(lora_layer)
            blk.attn.qkv = lora_layer

        self.Lora_gpt2 = model

    def forward(self, x):
        return self.Lora_gpt2(inputs_embeds=x).last_hidden_state
