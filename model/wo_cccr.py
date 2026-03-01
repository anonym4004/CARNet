import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted


class CoreExtract(nn.Module):
    def __init__(self, d_model, d_core, n_heads):
        super(CoreExtract, self).__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_core_head = d_core // n_heads
        self.gen1 = nn.Linear(d_model, d_model)
        self.gen2 = nn.Linear(d_model, self.d_head*self.n_heads)
        self.gen3 = nn.Conv1d(in_channels=self.d_head*self.n_heads,
                              out_channels=self.d_core_head*self.n_heads,
                              kernel_size=1,
                              groups=self.n_heads)
        self.gen4 = nn.Linear(d_core, d_core)
        
    def forward(self, x):
        batch_size, channels, d_model = x.shape
        H = self.n_heads
        multihead = F.gelu(self.gen2(x)).view(batch_size,channels,H,-1)
        multihead_core = multihead.reshape(batch_size*channels, H*self.d_head, 1)
        multihead_core = F.gelu(self.gen3(multihead_core)).view(batch_size,channels,H,-1)
        combined_mean = multihead_core.reshape(batch_size,channels,H*self.d_core_head)
        combined_mean = self.gen4(combined_mean)
        
        
        return combined_mean


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_core, n_heads, seq_len, d_ff=None, dropout=0.1, activation="relu", **kwargs):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.core_extract = CoreExtract(d_model, d_core, n_heads)
        
        self.proj = nn.Linear(seq_len, d_model)
        self.gen4 = nn.Linear(d_model + d_model, d_model)
        self.gen5 = nn.Linear(d_model + d_core, d_model)
        self.gen6 = nn.Linear(d_model, d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cycle, attn_mask=None, tau=None, delta=None, **kwargs):
        
        batch_size, channels, d_series = x.shape

        # set FFN
        proj = self.proj(cycle)
        combined_mean = torch.cat([proj, x], -1)
        corez = F.gelu(self.gen4(F.gelu(combined_mean)))
        core = self.core_extract(corez)

        

        # stochastic pooling
        if self.training:
            ratio = F.softmax(core, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            core = torch.gather(core, 1, indices)
            core = core.repeat(1, channels, 1)
        else:
            weight = F.softmax(core, dim=1)
            core = torch.sum(core * weight, dim=1, keepdim=True).repeat(1, channels, 1)
        
        

        # mlp fusion
        combined_mean_cat = torch.cat([corez, core], -1)
        combined_mean_cat = F.gelu(self.gen5(combined_mean_cat))
        combined_mean_cat = self.gen6(combined_mean_cat)
        new_x = combined_mean_cat

        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cycle, x_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cycle, attn_mask=x_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.cycle_len = configs.cycle
        self.data = torch.nn.Parameter(torch.zeros(self.cycle_len, configs.n_vars), requires_grad=True)
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.use_norm = configs.use_norm
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model,
                    configs.d_core,
                    configs.n_heads,
                    configs.seq_len,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
        )

        # Decoder
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, cycle_index, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, seq_len, N = x_enc.shape
        gather_index = (cycle_index.view(-1, 1) + torch.arange(seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
        cycle = self.data[gather_index].permute(0,2,1)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, cycle)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, cycle_index, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, cycle_index, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]