from math import sqrt

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoTokenizer, Qwen2Tokenizer, Qwen2Model, Qwen2Config
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from utils.interval_forecasting_tools import gaussian_sample, negative_binomial_sample
from utils.masking import masked_standardize_3d
from .Distribution import Gaussian,NegativeBinomial


transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    """Flatten and project patch embeddings to target prediction window.

    This module takes patch embeddings from the LLM encoder output, flattens them,
    and projects to the desired output sequence length. Used for both forecasting
    and imputation tasks.

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
            target_window: Target output sequence length.
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
    """LLM-based encoder block for time series imputation and forecasting.

    This module integrates pre-trained Large Language Models (LLAMA variants, GPT2,
    BERT, QWEN) with time series data through a reprogramming approach. It supports
    both imputation (filling missing values) and probabilistic forecasting tasks.

    The architecture includes:
    1. Patch embedding: Converts time series into overlapping patches
    2. Vocabulary mapping: Projects LLM embeddings to reduced token space
    3. Multi-head attention: Aligns time series with LLM representations
    4. LLM forward pass: Processes reprogrammed embeddings through frozen LLM
    5. Dynamic prompts: Generates statistical prompts for context (optional)
    6. Likelihood layers: Outputs distribution parameters (mu, sigma) for probabilistic forecasting

    Supports multiple LLM variants:
    - LLAMA8b (4096-dim), LLAMA3b (3072-dim), LLAMA1b (2048-dim)
    - GPT2 (768-dim), BERT (768-dim), QWEN (varies)

    Attributes:
        device: Computation device (cuda/cpu).
        task_name: Task type (forecasting/imputation).
        pred_len: Prediction horizon length.
        seq_len: Input sequence length.
        d_ff: Feed-forward dimension.
        d_llm: LLM hidden dimension.
        patch_len: Length of each patch.
        stride: Stride between consecutive patches.
        use_prompt: Whether to use dynamic text prompts.
        llm_model: Pre-trained LLM (frozen weights).
        tokenizer: LLM tokenizer for prompt processing.
        description: Domain description for prompts.
        patch_embedding: Patch embedding layer.
        mapping_layer: Vocabulary size reduction layer.
        MAtt: Multi-head attention for reprogramming.
        likelihood_layer_mu: Output layer for mean prediction.
        likelihood_layer_sigma: Output layer for standard deviation prediction.
        normalize_layers: Instance normalization layer.
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """Initialize the LLMBlock with a pre-trained language model.

        Loads and configures the specified LLM, freezes its parameters,
        and sets up reprogramming layers and likelihood outputs.

        Args:
            configs: Configuration object containing:
                - device: Computation device
                - task_name: Task type (forecasting/imputation)
                - pred_len: Prediction horizon
                - seq_len: Input sequence length
                - d_ff: Feed-forward dimension
                - d_model: Model embedding dimension
                - llm_model: LLM type ('LLAMA8b', 'LLAMA3b', 'LLAMA1b', 'GPT2', 'BERT', 'QWEN')
                - llm_dim: LLM hidden dimension
                - llm_layers: Number of LLM layers to use
                - llm_path: Path to LLM weights (for GPT2)
                - patch_len: Patch length for embedding
                - stride: Stride between patches
                - n_heads: Number of attention heads
                - dropout: Dropout probability
                - enc_in: Number of input features
                - use_prompt: Whether to use text prompts
                - prompt_domain: Whether to use domain-specific prompts
                - content: Domain description text
                - likelihood: Likelihood type ('g' for Gaussian, 'nb' for Negative Binomial)
            patch_len: Default patch length (overridden by configs).
            stride: Default stride (overridden by configs).
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

        if configs.llm_model == 'LLAMA8b':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained(r'D:\LLM\llama8b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    r'D:\LLM\llama8b',
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
                    r'D:\LLM\llama8b',
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
        elif configs.llm_model == 'LLAMA1b':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained(r'D:\LLM\llama1b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    r'D:\LLM\llama1b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-1b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    r'D:\LLM\llama1b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-1b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'LLAMA3b':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained(r'D:\LLM\llama3b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    r'D:\LLM\llama3b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-3b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    r'D:\LLM\llama3b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-3b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained(configs.llm_path)

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    configs.llm_path,
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
                    configs.llm_path,
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
            self.bert_config = Qwen2Config.from_pretrained(r'D:\LLM\qwen')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = Qwen2Model.from_pretrained(
                    r'D:\LLM\qwen',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = Qwen2Model.from_pretrained(
                    'Qwen/Qwen2-7B-Instruct',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = Qwen2Tokenizer.from_pretrained(
                    r'D:\LLM\qwen',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = Qwen2Tokenizer.from_pretrained(
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

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 2000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.MAtt = MultiheadAttention(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if configs.likelihood == "g":
            # self.likelihood_layer = Gaussian(configs.d_model, configs.c_out)
            self.likelihood_layer_mu = FlattenHead(configs.enc_in, self.head_nf, self.pred_len, head_dropout=configs.dropout)
            self.likelihood_layer_sigma = FlattenHead(configs.enc_in, self.head_nf, self.pred_len, head_dropout=configs.dropout)
        elif configs.likelihood == "nb":
            self.likelihood_layer = NegativeBinomial(configs.d_model, configs.c_out)
        else:
            self.likelihood_layer = Gaussian(configs.d_model, configs.c_out)

        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        self.llm_model.to(device=self.device)
        self.mapping_layer.to(device=self.device)
        self.MAtt.to(device=self.device)
        self.patch_embedding.to(device=self.device)

    def forward(self, x_enc, mask=None):
        """Forward pass through the LLMBlock.

        Args:
            x_enc: Encoder input tensor of shape (batch, seq_len, n_features).
            mask: Optional mask tensor (unused in this implementation).

        Returns:
            Forecast output tensor from interval_forecast method.
        """
        enc_out = self.interval_forecast(x_enc)
        return enc_out

    def imputation_forecast(self, x_enc, missing_mask):
        """Perform imputation forecasting with dynamic prompts.

        Generates text prompts with input statistics (min, max, median, trend, lags)
        and uses them to guide the LLM for imputation of missing values.

        The pipeline:
        1. Normalize input
        2. Compute per-variable statistics (min, max, median, trend, lags)
        3. Generate dynamic text prompts with statistics
        4. Convert time series to patches
        5. Reprogram patches to LLM space
        6. Concatenate prompt embeddings with patch embeddings
        7. Process through LLM
        8. Project to output and denormalize

        Args:
            x_enc: Encoder input of shape (batch, seq_len, n_features).
            missing_mask: Boolean mask indicating missing values.

        Returns:
            Imputed forecast tensor of shape (batch, pred_len, n_features).
        """
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        # Reshape for per-variable processing
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # Compute statistics for prompt generation
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        # Generate dynamic prompts with statistics
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information with missing values; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top {self.top_k} lags are : {lags_values_str}<|end_prompt|>"
            )

            prompt.append(prompt_)

        # Reshape back for embedding
        x_enc = x_enc.reshape(B, N, T)

        # Tokenize and embed prompts
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))

        # Project LLM vocabulary to reduced token space
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        # Create patch embeddings and reprogram to LLM space
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        # Concatenate prompt and patch embeddings
        if self.use_prompt:
            llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        else:
            llama_enc_out = enc_out

        # Process through LLM
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        # Reshape and project to output
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # Denormalize output
        dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out

    def interval_forecast(self, x_enc):
        """Perform probabilistic interval forecasting.

        Generates predictions with optional dynamic prompts containing
        comprehensive input statistics (mean, std, min, max, median, trend, lags).

        The pipeline:
        1. Normalize input
        2. If using prompts: compute statistics and generate text prompts
        3. Convert time series to patches
        4. Reprogram patches to LLM space via multi-head attention
        5. Optionally concatenate prompt embeddings
        6. Process through frozen LLM
        7. Project to output and denormalize

        Args:
            x_enc: Encoder input of shape (batch, seq_len, n_features).

        Returns:
            Forecast tensor of shape (batch, pred_len, n_features).
        """
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        if self.use_prompt:
            # Reshape for per-variable statistics computation
            x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

            # Compute comprehensive statistics
            min_values = torch.min(x_enc, dim=1)[0]
            max_values = torch.max(x_enc, dim=1)[0]
            medians = torch.median(x_enc, dim=1).values
            mean = torch.mean(x_enc, dim=1)
            std = torch.std(x_enc, dim=1)
            lags = self.calcute_lags(x_enc)
            trends = x_enc.diff(dim=1).sum(dim=1)

            # Generate dynamic prompts with all statistics
            prompt = []
            for b in range(x_enc.shape[0]):
                mean_values_str = str(mean[b].tolist()[0])
                std_values_str = str(std[b].tolist()[0])
                min_values_str = str(min_values[b].tolist()[0])
                max_values_str = str(max_values[b].tolist()[0])
                median_values_str = str(medians[b].tolist()[0])
                lags_values_str = str(lags[b].tolist())
                prompt_ = (
                    f"<|start_prompt|>Dataset description: {self.description}"
                    f"Task description: probabilistic forecast the next {str(self.pred_len)} steps mu and sigma given the previous {str(self.seq_len)} steps information; "
                    "Input statistics: "
                    f"mean value {mean_values_str}, "
                    f"standard deviation value {std_values_str}, "
                    f"min value {min_values_str}, "
                    f"max value {max_values_str}, "
                    f"median value {median_values_str}, "
                    f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                    f"top {self.top_k} lags are : {lags_values_str}<|end_prompt|>"
                )
                prompt.append(prompt_)

            # Reshape back
            x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

            # Tokenize and embed prompts
            prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))

        # Project LLM vocabulary to reduced token space
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        # Create patch embeddings
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))

        # Reprogram to LLM space via multi-head attention
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        # Concatenate prompt embeddings if using prompts
        if self.use_prompt:
            llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        else:
            llama_enc_out = enc_out

        # Process through frozen LLM
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        # Reshape and project to output
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # Denormalize output
        dec_out = self.normalize_layers(dec_out, 'denorm')

        # Note: Probabilistic output (mu, sigma) computation is available but commented out
        # mu = self.likelihood_layer_mu(dec_out[:, :, :, -self.patch_nums:])
        # sigma = torch.log(1 + torch.exp(self.likelihood_layer_sigma(dec_out[:, :, :, -self.patch_nums:]))) + 1e-6
        # mu = self.normalize_layers(mu.permute(0, 2, 1).contiguous(), 'denorm')
        # sigma = sigma.permute(0, 2, 1).contiguous() * self.normalize_layers.stdev
        return dec_out

    def calcute_lags(self, x_enc):
        """Calculate top-k autocorrelation lags using FFT.

        Computes autocorrelation via the Wiener-Khinchin theorem using FFT,
        then identifies the top-k lag values with highest correlation.
        Used for generating statistical prompts.

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


class MultiheadAttention(nn.Module):
    """Multi-head attention layer for reprogramming time series to LLM token space.

    This layer implements the reprogramming mechanism that aligns time series
    patch embeddings with LLM word embeddings using multi-head scaled dot-product
    attention. Queries come from time series patches while keys and values come
    from LLM word embeddings.

    Attributes:
        query_projection: Linear layer projecting time series patches to query space.
        key_projection: Linear layer projecting LLM embeddings to key space.
        value_projection: Linear layer projecting LLM embeddings to value space.
        out_projection: Linear layer projecting attention output to LLM dimension.
        n_heads: Number of attention heads.
        dropout: Dropout layer for attention weights.
    """

    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        """Initialize the MultiheadAttention layer.

        Args:
            d_model: Dimension of time series patch embeddings.
            n_heads: Number of attention heads.
            d_keys: Dimension of keys per head (default: d_model // n_heads).
            d_llm: Dimension of LLM embeddings.
            attention_dropout: Dropout probability for attention weights.
        """
        super(MultiheadAttention, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        """Forward pass computing multi-head attention for reprogramming.

        Args:
            target_embedding: Time series patch embeddings (batch, num_patches, d_model).
            source_embedding: LLM word embeddings for keys (num_tokens, d_llm).
            value_embedding: LLM word embeddings for values (num_tokens, d_llm).

        Returns:
            Reprogrammed embeddings of shape (batch, num_patches, d_llm).
        """
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        # Project to multi-head query/key/value
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        # Compute reprogramming attention
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        """Compute scaled dot-product attention for reprogramming.

        Uses Einstein summation for efficient multi-head attention computation.
        The attention aligns each time series patch with relevant LLM tokens,
        enabling the model to leverage LLM knowledge for time series tasks.

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
    """LLMformer: Hybrid LLM-Transformer model for time series imputation and forecasting.

    This model combines a pre-trained Large Language Model (LLM) with a flexible
    encoder architecture (Transformer, LSTM, or Linear) for multivariate time series
    tasks. It uses a dual-encoder approach:
    1. LLM Encoder: Processes target variables through reprogrammed LLM
    2. Configurable Encoder: Processes covariate variables (Transformer/LSTM/Linear)

    The model supports probabilistic forecasting with Gaussian or Negative Binomial
    likelihood layers, outputting mean (mu) and standard deviation (sigma) for
    interval predictions.

    Attributes:
        pred_len: Prediction horizon length.
        seq_len: Input sequence length.
        label_len: Label prefix length for decoder.
        c_out: Number of target output variables.
        output_attention: Whether to output attention weights.
        output_ori: Whether to output original values along with imputed.
        likelihood: Likelihood type ('g' for Gaussian, 'nb' for Negative Binomial).
        use_norm: Whether to use instance normalization.
        encoder_other_model: Type of covariate encoder ('Transformer', 'LSTM', 'Linear').
        encoder_other: Covariate encoder module.
        LLM_encoder: LLM-based encoder for target features.
        dec_embedding: Decoder input embedding layer.
        decoder: Transformer decoder with self and cross attention.
        projection: Final linear projection to output dimension.
        likelihood_layer: Probabilistic output layer (Gaussian or NegativeBinomial).
    """

    def __init__(self, configs, encoder_other_model='Transformer'):
        """Initialize the LLMformer model.

        Args:
            configs: Configuration object containing model hyperparameters:
                - pred_len: Prediction horizon
                - seq_len: Input sequence length
                - label_len: Label prefix length
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
                - output_ori: Whether to output original values
                - likelihood: Likelihood type ('g' or 'nb')
                - use_norm: Whether to use normalization
                - rnn_dim: RNN hidden dimension (if using LSTM)
                - rnn_layers: Number of RNN layers (if using LSTM)
            encoder_other_model: Type of encoder for covariates
                ('Transformer', 'LSTM', or 'Linear').
        """
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.c_out = configs.c_out
        self.output_attention = configs.output_attention
        self.output_ori = configs.output_ori
        self.likelihood = configs.likelihood
        self.use_norm = configs.use_norm

        # Covariate Encoder: configurable architecture for non-target features
        self.encoder_other_model = encoder_other_model
        transformer_d_model = 64
        transformer_d_ff = transformer_d_model * 4
        transformer_enc_layers = configs.e_layers

        if self.encoder_other_model == 'Linear':
            self.encoder_other = nn.Linear(configs.enc_in, transformer_d_model)

        elif self.encoder_other_model == 'LSTM':
            self.encoder_other = nn.LSTM(configs.enc_in, configs.rnn_dim, num_layers=configs.rnn_layers, batch_first=True)

        elif self.encoder_other_model == 'Transformer':
            self.enc_embedding = DataEmbedding(configs.enc_in - configs.c_out, transformer_d_model, configs.embed, configs.freq, configs.dropout)

            self.encoder_other = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False), transformer_d_model, configs.n_heads),
                        transformer_d_model,
                        transformer_d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(transformer_enc_layers)
                ],
                norm_layer=torch.nn.LayerNorm(transformer_d_model)
            )

        # LLM Encoder: processes target features through pre-trained LLM
        self.LLM_encoder = LLMBlock(configs)

        # Decoder: fuses LLM and covariate encoder outputs
        self.dec_embedding = DataEmbedding(configs.c_out, transformer_d_model, configs.embed, configs.freq, configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        transformer_d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        transformer_d_model, configs.n_heads),
                    transformer_d_model,
                    transformer_d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(transformer_d_model),
        )

        # Output projections
        self.projection = nn.Linear(transformer_d_model, configs.c_out)

        # Likelihood layer for probabilistic output
        if configs.likelihood == "g":
            self.likelihood_layer = Gaussian(transformer_d_model, configs.c_out)
        elif configs.likelihood == "nb":
            self.likelihood_layer = NegativeBinomial(transformer_d_model, configs.c_out)
        else:
            self.likelihood_layer = Gaussian(transformer_d_model, configs.c_out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast):
        """Perform point forecasting with instance normalization.

        Applies non-stationary transformer normalization, processes through
        dual encoders, and denormalizes the output.

        Args:
            x_enc: Encoder input of shape (batch, seq_len, enc_in).
            x_mark_enc: Encoder time features of shape (batch, seq_len, time_dim).
            x_dec: Decoder input of shape (batch, label_len + pred_len, dec_in).
            x_mark_dec: Decoder time features.
            x_forecast: Optional external forecast features.

        Returns:
            Point forecast tensor of shape (batch, label_len + pred_len, c_out).
        """
        # Non-stationary Transformer normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Optional: incorporate external forecast features
        if self.use_forecast:
            means_forecast = x_forecast.mean(1, keepdim=True).detach()
            x_enc_forecast = x_forecast - means_forecast
            stdev_forecast = torch.sqrt(torch.var(x_enc_forecast, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_forecast /= stdev_forecast
            x_forecast_ = self.forecast_projection(x_forecast[:,-self.pred_len:,:])
            x_enc = torch.cat((x_enc, x_forecast_), dim=1)

        # Split target and covariate features
        x_enc_target = x_enc[:,:,-self.c_out:]

        # Encode target through LLM
        enc_out_target = self.LLM_encoder(x_enc_target)

        # Encode covariates through configurable encoder
        if self.encoder_other_model == 'Transformer':
            enc_in = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder_other(enc_in)
        elif self.encoder_other_model == 'LSTM':
            enc_out, (_) = self.encoder_other(x_enc)
            enc_out = self.encoder_linear_projection(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.encoder_other_model == 'Linear':
            enc_out = self.encoder_other(x_enc)
            enc_out = self.encoder_linear_projection(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # Decode: LLM output as query, covariate encoding as context
        dec_in = enc_out_target
        dec_in = self.dec_embedding(dec_in, x_mark_dec[:,-self.label_len-self.pred_len:,:])
        dec_out = self.decoder(dec_in, enc_out, x_mask=None, cross_mask=None)
        dec_out = self.projection(dec_out)

        # Denormalize output
        dec_out = dec_out * (stdev[:, :1, -self.c_out:].repeat(1, self.pred_len+self.label_len, 1))
        dec_out = dec_out + (means[:, :1, -self.c_out:].repeat(1, self.pred_len+self.label_len, 1))

        return dec_out

    def interval_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast):
        """Perform probabilistic interval forecasting.

        Similar to forecast() but outputs distribution parameters (mu, sigma)
        for interval predictions using the likelihood layer.

        Args:
            x_enc: Encoder input of shape (batch, seq_len, enc_in).
            x_mark_enc: Encoder time features.
            x_dec: Decoder input.
            x_mark_dec: Decoder time features.
            x_forecast: Optional external forecast features.

        Returns:
            Tuple of (dec_out, mu, sigma):
                - dec_out: Decoder hidden output
                - mu: Predicted mean of shape (batch, seq_len + pred_len, c_out)
                - sigma: Predicted std of shape (batch, seq_len + pred_len, c_out)
        """
        if self.use_norm:
            # Non-stationary Transformer normalization
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Optional: incorporate external forecast features
        if self.use_forecast:
            means_forecast = x_forecast.mean(1, keepdim=True).detach()
            x_enc_forecast = x_forecast - means_forecast
            stdev_forecast = torch.sqrt(torch.var(x_enc_forecast, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_forecast /= stdev_forecast
            x_forecast_ = self.forecast_projection(x_forecast[:,-self.pred_len:,:])
            x_enc = torch.cat((x_enc, x_forecast_), dim=1)

        # Split target features
        x_enc_target = x_enc[:,:,-self.c_out:]

        # Encode target through LLM
        enc_out_target = self.LLM_encoder(x_enc_target)

        # Encode covariates through configurable encoder
        if self.encoder_other_model == 'Transformer':
            enc_in = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder_other(enc_in)
        elif self.encoder_other_model == 'LSTM':
            enc_out, (_) = self.encoder_other(x_enc)
            enc_out = self.encoder_linear_projection(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.encoder_other_model == 'Linear':
            enc_out = self.encoder_other(x_enc)
            enc_out = self.encoder_linear_projection(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # Decode and compute likelihood parameters
        dec_in = enc_out_target
        dec_in = self.dec_embedding(dec_in, x_mark_dec)
        dec_out = self.decoder(dec_in, enc_out, x_mask=None, cross_mask=None)
        dec_out, mu, sigma = self.likelihood_layer(dec_out)

        # Denormalize outputs
        output_len = self.pred_len + self.seq_len
        if self.use_norm:
            mu = mu * (stdev[:, :1, -self.c_out:].repeat(1, output_len, 1))
            mu = mu + (means[:, :1, -self.c_out:].repeat(1, output_len, 1))
            sigma = sigma * (stdev[:, :1, -self.c_out:].repeat(1, output_len, 1))
            dec_out = dec_out * (stdev[:, :1, -self.c_out:].repeat(1, self.pred_len+self.seq_len, 1))
            dec_out = dec_out + (means[:, :1, -self.c_out:].repeat(1, self.pred_len+self.seq_len, 1))

        return dec_out, mu, sigma

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast=None, mask=None):
        """Forward pass for the LLMformer model.

        Performs probabilistic interval forecasting and returns predictions
        for the prediction horizon only.

        Args:
            x_enc: Encoder input of shape (batch, seq_len, enc_in).
            x_mark_enc: Encoder time features.
            x_dec: Decoder input.
            x_mark_dec: Decoder time features.
            x_forecast: Optional external forecast features.
            mask: Optional mask tensor (unused).

        Returns:
            Tuple of (dec_out, mu, sigma) for the prediction horizon:
                - dec_out: Predictions of shape (batch, pred_len, c_out)
                - mu: Predicted mean of shape (batch, pred_len, c_out)
                - sigma: Predicted std of shape (batch, pred_len, c_out)
        """
        dec_out, mu, sigma = self.interval_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast)
        return dec_out[:, -self.pred_len:, :], mu[:, -self.pred_len:, :], sigma[:, -self.pred_len:, :]
