import torch.nn as nn
import math

# LoRA: W + W' = W + BA
class _LoRA_qkv(nn.Module):
    """In Sam, it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q,
            linear_b_q,
            linear_a_v,
            linear_b_v,
            dim,
    ):
        super().__init__()
        self.qkv = qkv
        # LoRA: adapting both Wq ang Wv yields the best results
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = dim

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = x @ self.linear_a_q.transpose(0, 1) @ self.linear_b_q.transpose(0, 1)
        new_v = x @ self.linear_a_v.transpose(0, 1) @ self.linear_b_v.transpose(0, 1)
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv

## LoRA for GPT-2
class LoRA_gpt2(nn.Module):
    """Applies low-rank adaptation to a GPT-2 model."""

    def __init__(self, model, r: int, gpt_dim=768):
        super(LoRA_gpt2, self).__init__()

        #self.w_As = []  # These are linear layers
        #self.w_Bs = []
        self.w_As = nn.ParameterList()  # 自动注册参数
        self.w_Bs = nn.ParameterList()
        

        # freeze image encoder first
        for i, (name, param) in enumerate(model.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for i, blk in enumerate(model.h):
            # do LoRA operation on all the attn.c_attn in each block
            w_qkv_linear = blk.attn.c_attn
            self.dim = gpt_dim

            w_a_linear_q = nn.Parameter(w_qkv_linear.weight.new_zeros(r, self.dim))
            w_b_linear_q = nn.Parameter(w_qkv_linear.weight.new_zeros(self.dim, r))
            w_a_linear_v = nn.Parameter(w_qkv_linear.weight.new_zeros(r, self.dim))
            w_b_linear_v = nn.Parameter(w_qkv_linear.weight.new_zeros(self.dim, r))
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(  # modify the qkv module of the sam encoder
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                dim=self.dim,
            )
        #Initializing parameters
        self.reset_parameters()
        #Return the model after doing the LoRA operation
        self.Lora_gpt2 = model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A, a=math.sqrt(5))
            # nn.init.normal_(w_A.weight, mean=0, std=0.01)
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B)

    def forward(self, x):
        return self.Lora_gpt2(inputs_embeds=x).last_hidden_state
