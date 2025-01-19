import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.c_out = configs.c_out
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if configs.rnn_model == 'GRU':
            self.rnn_layer = nn.GRU(configs.enc_in, configs.rnn_dim, configs.rnn_layers, batch_first=True)
        elif configs.rnn_model == 'LSTM':
            self.rnn_layer = nn.LSTM(configs.enc_in, configs.rnn_dim, configs.rnn_layers, batch_first=True)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.linear_predict = nn.Linear(configs.seq_len, configs.pred_len)
            # self.output_projection = nn.Linear(configs.rnn_dim, configs.c_out)
            self.output_projection = nn.Linear(configs.rnn_dim, configs.c_out)

        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.output_projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x_enc = x - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x,_ = self.rnn_layer(x_enc)
        x = self.linear_predict(x.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = self.output_projection(x)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
