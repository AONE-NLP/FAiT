import torch
import torch.nn as nn
from layers.RevIN import RevIN
from layers.Transformer_EncDec import Encoder_ori, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention_ablation,FullAttention_ori
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models ... """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=nn.GELU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # channels
        self.seq_len = configs.seq_len
        self.hidden_size = self.d_model = configs.d_model  # hidden_size
        self.d_ff = configs.d_ff  # d_ff

        # self.channel_independence = configs.channel_independence
        self.embed_size = configs.embed_size  # embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.valid_fre_points = int((self.seq_len + 1) / 2 + 0.5)
        self.encoder_fre_tran_ori = Encoder_ori(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention_ori(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            one_output=True,
            CKA_flag=configs.CKA_flag
        )
        self.fre_trans = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_model),
            self.encoder_fre_tran_ori,
            nn.Linear(self.d_model, self.seq_len * self.embed_size)
        )

        # for final output
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

        self.qkv = nn.Linear(self.embed_size, self.embed_size * 3, bias=False)

        self.dy_freq_2 = nn.Linear(self.embed_size, self.embed_size, bias=True)
        self.lf_gamma = nn.Parameter(1e-5 * torch.ones(self.embed_size), requires_grad=True)  # no decay

        self.dy_freq = nn.Linear(self.embed_size, self.embed_size, bias=True)
        self.hf_gamma = nn.Parameter(1e-5 * torch.ones(self.embed_size), requires_grad=True)  # no decay
        self.proj = nn.Linear(self.embed_size, self.embed_size)

        # Mlp，mlp_ratio = reweight_expansion_ratio
        self.reweight_expansion_ratio = 0.125
        self.group = self.embed_size//4
        self.num_filters = self.embed_size//2
        self.reweight = Mlp(self.embed_size,
                            mlp_ratio=self.reweight_expansion_ratio,
                            out_features=self.group * self.num_filters,
                            act_layer=nn.GELU,
                            drop=configs.dropout,
                            bias=False)

        # [F, D//G, filter_size]
        self.complex_weights = nn.Parameter(
            torch.randn(self.num_filters, self.embed_size // self.group, self.valid_fre_points) * 1e-5
        )
        trunc_normal_(self.complex_weights, std=1e-5)

    # dimension extension
    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        # x: [B, T, N] --> [B, N, T]
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        # B*N*T*1 x 1*D = B*N*T*D
        return x * embeddings

    def Fre_Trans(self, x):
        # [B, N, T, D]
        # x = x.transpose(-1, -2)
        B, N, T, D = x.shape
        assert T == self.seq_len

        if hasattr(self, 'dy_freq_2'): #低频权重计算
            dy_freq_lf = self.dy_freq_2(x).tanh_()

        if hasattr(self, 'dy_freq'):
            dy_freq = F.softplus(self.dy_freq(x))
            dy_freq2 = dy_freq ** 2
            # dy_freq = dy_freq2 / (dy_freq2 + 1)
            dy_freq = 2 * dy_freq2 / (dy_freq2 + 0.3678)


        qkv = self.qkv(x).reshape(B, N, T, D, 3).permute(4, 0, 1, 3, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = self.fre_trans(qkv.flatten(-2)).reshape(B, N, D, T)
        v_hf = v - x

        # B, N, T, D
        x = x.transpose(-1, -2)
        v_hf = v_hf.transpose(-1, -2)

        if hasattr(self, 'dy_freq_2'):
            x = x + x * dy_freq_lf * self.lf_gamma.view(1, 1, 1, -1)
        if hasattr(self, 'dy_freq'):
            x = x + dy_freq * v_hf * self.hf_gamma.view(1, 1, 1, -1)

        x = self.dropout(self.proj(x))
        # B, N, D, T
        x = x.transpose(-1, -2)
        x = x.reshape(B * N, D, T)
        # FFT along T
        x_rfft = torch.fft.rfft(x, dim=-1, norm='ortho')  # [BN, D, T//2+1]

        # global mean pooling -> routing
        route_in = x.mean(dim=-1)  # [BN, D]
        routeing = self.reweight(route_in).view(B * N, self.group, self.num_filters).tanh_()  # [BN, G, F]

        # interpolate weights if T changed
        weight = self.complex_weights  # [F, D//G, f_len0]
        if weight.shape[-1] != x_rfft.shape[-1]:
            weight = F.interpolate(
                weight.unsqueeze(0),
                size=x_rfft.shape[-1],
                mode='linear',
                align_corners=False
            ).squeeze(0)  # [F, D//G, f_len1]

        # routing-weighted filters: [BN, G, D//G, f_len]
        weight = torch.einsum('bgf,fdl->bgdl', routeing, weight)
        weight = weight.reshape(B * N, D, x_rfft.shape[-1])  # [BN, D, f_len]

        # complex mul (real & imag share the same real weight)
        x_rfft = torch.view_as_complex(
            torch.stack([x_rfft.real * weight,
                         x_rfft.imag * weight], dim=-1)
        )

        # IFFT
        x = torch.fft.irfft(x_rfft, n=T, dim=-1, norm='ortho')  # [BN, D, T]
        x = x.reshape(B, N, D, T)

        # [B, N, T, D]
        x = x.transpose(-1, -2)
        return x

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape #32 96 7

        # revin norm
        x = self.revin_layer(x, mode='norm') #32 96 7

        # ###########  frequency (high-level) part ##########
        # input fre fine-tuning
        # [B, T, N]
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x, self.embeddings)  # self.embeddings 1 8   x 32 7 96 8
        # [B, N, T, D]
        x = self.Fre_Trans(x) + x

        # linear
        # [B, N, T*D] --> [B, N, dim] --> [B, N, tau] --> [B, tau, N]
        out = self.fc(x.flatten(-2)).transpose(-1, -2)

        # dropout
        out = self.dropout(out)

        # revin denorm
        out = self.revin_layer(out, mode='denorm')

        return out
