import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from utils.interval_forecasting_tools import gaussian_sample, negative_binomial_sample
from .Distribution import Gaussian, NegativeBinomial


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.likelihood = configs.likelihood
        self.c_out = configs.c_out
        self.use_norm = configs.use_norm
        self.output_ori = configs.output_ori

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.use_forecast = configs.use_forecast
        if self.use_forecast:
            self.forecast_projection = nn.Linear(configs.forecast_dim, configs.enc_in)

        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'interval_forecast' or self.task_name == 'imputation_forecast':

            self.dec_embedding = DataEmbedding(configs.c_out, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
            )
            # self.decoder = nn.Linear(configs.d_model, configs.c_out)
            # self.linear_projection = nn.Linear(configs.seq_len, configs.pred_len)
            self.output_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

            if self.use_forecast:
                self.linear_projection = nn.Linear(configs.seq_len + configs.pred_len, configs.pred_len)
                self.dec_embedding = DataEmbedding(configs.forecast_dim, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)
        if self.task_name == 'interval_forecast':
            if configs.likelihood == "g":
                self.likelihood_layer = Gaussian(configs.d_model, configs.c_out)
            elif configs.likelihood == "nb":
                self.likelihood_layer = NegativeBinomial(configs.d_model, configs.c_out)
            else:
                self.likelihood_layer = Gaussian(configs.d_model, configs.c_out)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast=None,mask=None):
        if mask is None:
            mask = torch.ones_like(x_enc)
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
            means = means.unsqueeze(1).detach()
            x_enc = x_enc - means
            # x_enc = x_enc.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(1).detach()
            x_enc /= stdev
        # if self.use_forecast:
        #     x_forecast_ = self.forecast_projection(x_forecast)
        #     x_enc = torch.cat((x_enc, x_forecast_), dim=1)
        # x_mark_enc = x_mark_dec
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_forecast, x_mark_dec) if self.use_forecast else self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        dec_out = self.output_projection(dec_out)
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            # if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            #     # dec_out = self.linear_projection(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
            #     dec_out = dec_out * (stdev[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.pred_len, 1))
            #     dec_out = dec_out + (means[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.pred_len, 1))
            # else:
            dec_out = dec_out * (stdev[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
            dec_out = dec_out + (means[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def interval_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        # if self.use_forecast:
        #     x_forecast_ = self.forecast_projection(x_forecast)
        #     x_enc = torch.cat((x_enc, x_forecast_), dim=1)
        # x_mark_enc = x_mark_dec
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_forecast, x_mark_dec) if self.use_forecast else self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out, mu, sigma = self.likelihood_layer(dec_out)

        if self.use_norm:
            mu = mu * (stdev[:, :1, -self.c_out:].repeat(1, self.seq_len + self.pred_len, 1))
            mu = mu + (means[:, :1, -self.c_out:].repeat(1, self.seq_len + self.pred_len, 1))
            sigma = sigma * (stdev[:, :1, -self.c_out:].repeat(1, self.seq_len + self.pred_len, 1))
            dec_out = dec_out * (stdev[:, :1, -self.c_out:].repeat(1, self.seq_len + self.pred_len, 1))
            dec_out = dec_out + (means[:, :1, -self.c_out:].repeat(1, self.seq_len + self.pred_len, 1))
        return dec_out, mu, sigma

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        # output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast, mask)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast, mask)
            if self.output_ori:
                dec_out[:, :self.seq_len, -self.c_out:] = mask[:, :, -self.c_out:] * x_enc[:, :self.seq_len, -self.c_out:] + (1 - mask[:, :, -self.c_out:]) * dec_out[:, :self.seq_len, -self.c_out:]
            return dec_out  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            if self.output_ori:
                dec_out = mask[:, :, -self.c_out:] * x_enc[:, :, -self.c_out:] + (1 - mask[:, :, -self.c_out:]) * dec_out
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == 'interval_forecast':
            dec_out, mu, sigama = self.interval_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast)
            return dec_out[:, -self.pred_len:, :], mu[:, -self.pred_len:, :], sigama[:, -self.pred_len:, :]
        return None
