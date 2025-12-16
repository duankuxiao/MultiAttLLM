from math import sqrt

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    """Flatten and project patch embeddings to target prediction window.

    This module takes patch embeddings from the LLM encoder output, flattens them,
    and projects to the desired output sequence length for forecasting.

    Attributes:
        n_vars: Number of variables/features in the time series.
        flatten: Flatten layer to merge patch and feature dimensions.
        linear: Linear projection to target window size.
        dropout: Dropout layer for regularization.
    """

    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        """Initialize the FlattenHead module.

        Args:
            n_vars: Number of variables in the time series.
            nf: Input feature dimension (d_ff * patch_nums).
            target_window: Target output sequence length (pred_len + label_len).
            head_dropout: Dropout probability for regularization.
        """
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """Forward pass through the flatten head.

        Args:
            x: Input tensor of shape (batch, n_vars, d_ff, patch_nums).

        Returns:
            Output tensor of shape (batch, n_vars, target_window).
        """
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class LLMBlock(nn.Module):
    """LLM-based encoder block for time series forecasting.

    This module integrates pre-trained Large Language Models (LLAMA, GPT2, BERT, QWEN)
    with time series data through a reprogramming approach. It converts time series
    patches into LLM token space using cross-attention, leveraging the LLM's
    pre-trained knowledge for enhanced forecasting.

    The architecture consists of:
    1. Patch embedding: Converts time series into overlapping patches
    2. Word projection: Maps LLM vocabulary embeddings to a reduced token set
    3. Cross-attention: Aligns time series patches with LLM token embeddings
    4. LLM forward pass: Processes reprogrammed embeddings through frozen LLM layers
    5. Output projection: Maps LLM outputs back to prediction space

    Attributes:
        device: Device for computation (cuda/cpu).
        task_name: Name of the task (forecasting/imputation).
        pred_len: Prediction horizon length.
        seq_len: Input sequence length.
        d_ff: Feed-forward dimension for output projection.
        top_k: Number of top lags to consider for autocorrelation.
        d_llm: LLM hidden dimension size.
        patch_len: Length of each patch.
        stride: Stride between consecutive patches.
        use_prompt: Whether to use text prompts.
        llm_model: Pre-trained LLM model (frozen weights).
        tokenizer: Tokenizer for the LLM.
        patch_embedding: Patch embedding layer.
        word_embeddings: LLM word embeddings.
        word_projection: Linear layer to reduce vocabulary size.
        crossattention_layer: Cross-attention for reprogramming.
        normalize_layers: Instance normalization layer.
        output_projection: Projects LLM output to prediction space.
    """

    def __init__(self, configs):
        """Initialize the LLMBlock with a pre-trained language model.

        Loads and configures the specified LLM (LLAMA, GPT2, BERT, or QWEN),
        freezes its parameters, and sets up the reprogramming layers for
        time series to LLM token space alignment.

        Args:
            configs: Configuration object containing:
                - device: Computation device (cuda/cpu)
                - task_name: Task type (forecasting/imputation)
                - pred_len: Prediction horizon length
                - seq_len: Input sequence length
                - d_ff: Feed-forward dimension
                - d_model: Model embedding dimension
                - llm_model: LLM type ('LLAMA', 'GPT2', 'BERT', 'QWEN')
                - llm_dim: LLM hidden dimension
                - llm_layers: Number of LLM layers to use
                - patch_len: Patch length for embedding
                - stride: Stride between patches
                - n_heads: Number of attention heads
                - dropout: Dropout probability
                - enc_in: Number of input features
                - label_len: Label prefix length
                - use_prompt: Whether to use text prompts
                - use_forecast: Whether in forecasting mode
        """
        super(LLMBlock, self).__init__()
        self.device = configs.device
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = configs.top_k
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        self.use_prompt = configs.use_prompt
        self.use_forecast = configs.use_forecast

        # Load the specified pre-trained LLM and tokenizer
        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained(r'D:\LLM\llama')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    r'D:\LLM\llama',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    r'D:\LLM\llama',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained(r'D:\LLM\gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    r'D:\LLM\gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    r'D:\LLM\gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'QWEN':
            self.bert_config = BertConfig.from_pretrained(r'D:\LLM\qwen')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    r'D:\LLM\qwen',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'Qwen/Qwen2-7B-Instruct',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    r'D:\LLM\qwen',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'Qwen/Qwen2-7B-Instruct',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained(r'D:\LLM\bert')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    r'D:\LLM\bert',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    r'D:\LLM\bert',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(configs.dropout)
        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, configs.dropout)
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 3000
        self.word_projection = nn.Linear(self.vocab_size, self.num_tokens)
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len + configs.label_len, head_dropout=configs.dropout)

        self.crossattention_layer = CrossAttentionLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        self.llm_model.to(device=self.device)
        self.word_projection.to(device=self.device)
        self.crossattention_layer.to(device=self.device)
        self.patch_embedding.to(device=self.device)
        self.output_projection.to(device=self.device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass through the LLMBlock.

        Args:
            x_enc: Encoder input tensor of shape (batch, seq_len, n_features).
            x_mark_enc: Encoder time features (unused in this implementation).
            x_dec: Decoder input tensor (unused in this implementation).
            x_mark_dec: Decoder time features (unused in this implementation).
            mask: Optional mask tensor (unused in this implementation).

        Returns:
            Forecast output tensor of shape (batch, pred_len + label_len, n_features).
        """
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,)
        return dec_out[:, :, :]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Perform forecasting using LLM-based reprogramming.

        The forecasting pipeline:
        1. Normalize input using instance normalization
        2. Project LLM word embeddings to reduced token space
        3. Convert time series to patch embeddings
        4. Reprogram patches to LLM space via cross-attention
        5. Process through frozen LLM layers
        6. Project output back to prediction space
        7. Denormalize output

        Args:
            x_enc: Encoder input of shape (batch, seq_len, n_features).
            x_mark_enc: Encoder time features (unused).
            x_dec: Decoder input (unused).
            x_mark_dec: Decoder time features (unused).

        Returns:
            Forecast tensor of shape (batch, pred_len + label_len, n_features).
        """
        # Step 1: Instance normalization
        x_enc = self.normalize_layers(x_enc, 'norm')

        # Step 2: Project word embeddings to reduced vocabulary
        word_embeddings = self.word_projection(self.word_embeddings.permute(1, 0)).permute(1, 0)

        # Step 3: Convert to patches - reshape to (batch, n_vars, seq_len) for patching
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))

        # Step 4: Cross-attention reprogramming to LLM token space
        enc_out = self.crossattention_layer(enc_out, word_embeddings, word_embeddings)

        # Step 5: Forward pass through frozen LLM
        dec_out = self.llm_model(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        # Step 6: Reshape and project to output space
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # Step 7: Denormalize output
        dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out

    def calcute_lags(self, x_enc):
        """Calculate top-k autocorrelation lags using FFT.

        Computes autocorrelation via the Wiener-Khinchin theorem using FFT,
        then identifies the top-k lag values with highest correlation.

        Args:
            x_enc: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            Tensor of top-k lag indices with shape (batch, top_k).
        """
        # Compute autocorrelation using FFT (Wiener-Khinchin theorem)
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # Average across features and find top-k lags
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for reprogramming time series to LLM token space.

    This layer implements the reprogramming mechanism that aligns time series
    patch embeddings with LLM word embeddings. It uses scaled dot-product
    attention where queries come from time series patches and keys/values
    come from LLM word embeddings.

    Attributes:
        query_projection: Linear layer projecting time series patches to query space.
        key_projection: Linear layer projecting LLM embeddings to key space.
        value_projection: Linear layer projecting LLM embeddings to value space.
        out_projection: Linear layer projecting attention output to LLM dimension.
        n_heads: Number of attention heads.
        dropout: Dropout layer for attention weights.
    """

    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        """Initialize the CrossAttentionLayer.

        Args:
            d_model: Dimension of time series patch embeddings.
            n_heads: Number of attention heads.
            d_keys: Dimension of keys per head (default: d_model // n_heads).
            d_llm: Dimension of LLM embeddings.
            attention_dropout: Dropout probability for attention weights.
        """
        super(CrossAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        """Forward pass computing cross-attention between time series and LLM embeddings.

        Args:
            target_embedding: Time series patch embeddings (batch, num_patches, d_model).
            source_embedding: LLM word embeddings for keys (num_tokens, d_llm).
            value_embedding: LLM word embeddings for values (num_tokens, d_llm).

        Returns:
            Reprogrammed embeddings of shape (batch, num_patches, d_llm).
        """
        B, T, N = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        # Project to multi-head query/key/value
        target_embedding = self.query_projection(target_embedding).view(B, T, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        # Compute reprogramming attention
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, T, -1)
        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        """Compute scaled dot-product attention for reprogramming.

        Uses Einstein summation for efficient multi-head attention computation.
        The attention aligns each time series patch with relevant LLM tokens.

        Args:
            target_embedding: Query tensor (batch, num_patches, n_heads, head_dim).
            source_embedding: Key tensor (num_tokens, n_heads, head_dim).
            value_embedding: Value tensor (num_tokens, n_heads, head_dim).

        Returns:
            Attention output tensor (batch, num_patches, n_heads, head_dim).
        """
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        # Compute attention scores: (batch, heads, patches, tokens)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        # Apply softmax and dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # Weighted sum of values
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        return reprogramming_embedding


class Model(nn.Module):
    """MultiAttLLM: Multi-Attention LLM model for time series forecasting.

    This model combines a pre-trained Large Language Model (LLM) with a
    Transformer encoder-decoder architecture for multivariate time series
    forecasting. The architecture uses a dual-encoder approach:
    1. LLM Encoder: Processes target variables through reprogrammed LLM
    2. Transformer Encoder: Processes covariate (non-target) variables

    The decoder uses cross-attention to fuse information from both encoders,
    enabling the model to leverage both LLM knowledge and covariate patterns.

    Attributes:
        task_name: Name of the task (forecasting).
        pred_len: Prediction horizon length.
        c_out: Number of target output variables.
        output_attention: Whether to output attention weights.
        use_forecast: Whether in forecasting mode.
        enc_embedding: Embedding layer for covariate encoder input.
        encoder: Transformer encoder for covariate features.
        LLM_encoder: LLM-based encoder for target features.
        dec_embedding: Embedding layer for decoder input.
        selfattention_layer: Transformer decoder with self and cross attention.
        out_projection: Final linear projection to output dimension.
        linear_predict: Linear layer for sequence length transformation.
    """

    def __init__(self, configs):
        """Initialize the MultiAttLLM model.

        Args:
            configs: Configuration object containing model hyperparameters:
                - task_name: Task type identifier
                - pred_len: Prediction horizon
                - c_out: Number of output target variables
                - enc_in: Total number of input features
                - d_model: Model embedding dimension
                - n_heads: Number of attention heads
                - e_layers: Number of encoder layers
                - d_layers: Number of decoder layers
                - factor: Attention factor
                - dropout: Dropout probability
                - activation: Activation function type
                - embed: Embedding type
                - freq: Time frequency for temporal embedding
                - output_attention: Whether to output attention weights
                - use_forecast: Whether in forecasting mode
                - seq_len: Input sequence length
                - label_len: Label prefix length
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.output_attention = configs.output_attention
        self.use_forecast = configs.use_forecast

        # Covariate Encoder: processes non-target features
        self.enc_embedding = DataEmbedding(configs.enc_in-self.c_out, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    4*configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # LLM Encoder: processes target features through pre-trained LLM
        self.LLM_encoder = LLMBlock(configs)

        # Decoder: fuses LLM and covariate encoder outputs
        self.dec_embedding = DataEmbedding(configs.c_out, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.selfattention_layer = Decoder(
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
                    4 * configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # Output projections
        self.out_projection = nn.Linear(configs.d_model, configs.c_out)
        self.linear_predict = nn.Linear(configs.seq_len, configs.pred_len + configs.label_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Perform multivariate time series forecasting.

        Processes target and covariate features through separate encoders,
        then fuses them in the decoder for final prediction.

        Args:
            x_enc: Encoder input of shape (batch, seq_len, enc_in).
            x_mark_enc: Encoder time features of shape (batch, seq_len, time_dim).
            x_dec: Decoder input of shape (batch, label_len + pred_len, dec_in).
            x_mark_dec: Decoder time features of shape (batch, label_len + pred_len, time_dim).

        Returns:
            Forecast tensor of shape (batch, label_len + pred_len, c_out).
        """
        # Split input into covariate and target features
        x_enc_other = x_enc[:,:,:-self.c_out]
        x_enc_target = x_enc[:,:,-self.c_out:]

        # Encode target features through LLM
        enc_out_target = self.LLM_encoder(x_enc_target, x_mark_enc, x_dec, x_mark_dec)

        # Encode covariate features through Transformer
        enc_out_other, attn = self.encoder(self.enc_embedding(x_enc_other, x_mark_enc))

        # Decode: use LLM output as decoder input, cross-attend to covariate encoding
        dec_in = enc_out_target
        dec_in = self.dec_embedding(dec_in, x_mark_dec)
        dec_out = self.selfattention_layer(dec_in, enc_out_other, x_mask=None, cross_mask=None)

        # Project to output dimension
        dec_out = self.out_projection(dec_out)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass for the MultiAttLLM model.

        Args:
            x_enc: Encoder input of shape (batch, seq_len, enc_in).
            x_mark_enc: Encoder time features.
            x_dec: Decoder input.
            x_mark_dec: Decoder time features.
            mask: Optional mask tensor (unused).

        Returns:
            Forecast predictions of shape (batch, pred_len, c_out).
        """
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
