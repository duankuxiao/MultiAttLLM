import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
from utils.interval_forecasting_tools import gaussian_sample, negative_binomial_sample
from .Distribution import Gaussian,NegativeBinomial

class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.likelihood = configs.likelihood
        self.label_len = configs.label_len
        self.use_forecast = configs.use_forecast
        self.c_out = configs.c_out
        self.use_norm = configs.use_norm
        self.output_ori = configs.output_ori

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.c_out, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil and ('forecast' in configs.task_name) else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
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
        # self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)
        if self.task_name == 'interval_forecast':
            if configs.likelihood == "g":
                self.likelihood_layer = Gaussian(configs.d_model, configs.c_out)
            elif configs.likelihood == "nb":
                self.likelihood_layer = NegativeBinomial(configs.d_model, configs.c_out)
            else:
                self.likelihood_layer = Gaussian(configs.d_model, configs.c_out)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,mask=None):
        if mask is None:
            mask = torch.ones_like(x_enc)
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
            means = means.unsqueeze(1).detach()
            x_enc = x_enc - means
            x_enc = x_enc.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(1).detach()
            x_enc /= stdev
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            if self.task_name == 'imputation_forecast':
                dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
                dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
            else:
                dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
                dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out  # [B, L, D]

    def interval_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,x_forecast=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        if self.use_forecast:
            means_forecast = x_forecast.mean(1, keepdim=True).detach()
            x_enc_forecast = x_forecast - means_forecast
            stdev_forecast = torch.sqrt(torch.var(x_enc_forecast, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_forecast /= stdev_forecast
            x_forecast_ = self.forecast_projection(x_forecast[:, -self.pred_len:, :])
            x_enc = torch.cat((x_enc, x_forecast_), dim=1)
            x_mark_enc = x_mark_dec

        enc_in = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_in, attn_mask=None)
        dec_in = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_in, enc_out, x_mask=None, cross_mask=None)
        dec_out, mu, sigma = self.likelihood_layer(dec_out)
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            mu = mu * (stdev[:, :1, -self.c_out:].repeat(1, self.seq_len + self.pred_len, 1))
            mu = mu + (means[:, :1, -self.c_out:].repeat(1, self.seq_len + self.pred_len, 1))
            sigma = sigma * (stdev[:, :1, -self.c_out:].repeat(1, self.seq_len + self.pred_len, 1))
            dec_out = dec_out * (stdev[:, :1, -self.c_out:].repeat(1, self.seq_len + self.pred_len, 1))
            dec_out = dec_out + (means[:, :1, -self.c_out:].repeat(1, self.seq_len + self.pred_len, 1))

        return dec_out, mu, sigma  # [B, L, D]

    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        dec_out = self.projection(dec_out)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out  # [B, L, D]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
            means = means.unsqueeze(1).detach()
            x_enc = x_enc - means
            x_enc = x_enc.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(1).detach()
            x_enc /= stdev
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.seq_len, 1))
            dec_out = dec_out + (means[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        # output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,x_forecast=None, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            if self.output_ori:
                dec_out = mask[:, :, -self.c_out:] * x_enc[:, :, -self.c_out:] + (1 - mask[:, :, -self.c_out:]) * dec_out
            return dec_out  # [B, L, D]
        if self.task_name == 'imputation_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,mask)
            if self.output_ori:
                dec_out[:, :self.seq_len, -self.c_out:] = mask[:, :, -self.c_out:] * x_enc[:, :self.seq_len, -self.c_out:] + (1 - mask[:, :, -self.c_out:]) * dec_out[:, :self.seq_len, -self.c_out:]
            return dec_out
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