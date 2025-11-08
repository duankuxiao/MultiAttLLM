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
from utils.masking import masked_standardize_3d



transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class LLMBlock(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
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
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if 'forecast' in self.task_name:
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len + configs.label_len, head_dropout=configs.dropout)
            self.output_projection.to(device=self.device)
        if self.task_name == 'imputation':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.seq_len, head_dropout=configs.dropout)
            self.output_projection.to(device=self.device)

        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        self.llm_model.to(device=self.device)
        self.mapping_layer.to(device=self.device)
        self.reprogramming_layer.to(device=self.device)
        self.patch_embedding.to(device=self.device)

    def forward(self, x_enc, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, )
            return dec_out
        if self.task_name == 'imputation_forecast':
            dec_out = self.imputation_forecast(x_enc, mask)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, mask)
            return dec_out
        if self.task_name == 'interval_forecast':
            # mu, sigma = self.interval_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast)
            # return mu, sigma
            enc_out = self.interval_forecast(x_enc)
            return enc_out
        else:
            dec_out = self.forecast(x_enc)
            return dec_out
        return None

    def forecast(self, x_enc,):
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top {self.top_k} lags are : {lags_values_str}<|end_prompt|>"
            )
            # prompt_ = (
            #     f"<|start_prompt|>Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information <|end_prompt|> "
            # )

            # prompt_ = (
            #     f"<|start_prompt|>Dataset description: {self.description}"
            #     f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information <|end_prompt|> "
            # )

            prompt.append(prompt_)
        # x_enc [B * N, T, 1]
        x_enc = x_enc.reshape(B, N, T) # [B, T, N]

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        if self.use_prompt:
            llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        else:
            llama_enc_out = enc_out

        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out

    def imputation_forecast(self, x_enc, missing_mask):
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

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
        # x_enc [B * N, T, 1]
        x_enc = x_enc.reshape(B, N, T) # [B, T, N]

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        if self.use_prompt:
            llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        else:
            llama_enc_out = enc_out

        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out

    def imputation(self, x_enc, missing_mask):
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()
        if missing_mask is not None:
            missing_mask = missing_mask.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        if self.use_prompt:
            x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

            min_values = torch.min(x_enc, dim=1)[0]
            max_values = torch.max(x_enc, dim=1)[0]
            medians = torch.median(x_enc, dim=1).values
            lags = self.calcute_lags(x_enc)
            trends = x_enc.diff(dim=1).sum(dim=1)

            prompt = []
            for b in range(x_enc.shape[0]):
                min_values_str = str(min_values[b].tolist()[0])
                max_values_str = str(max_values[b].tolist()[0])
                median_values_str = str(medians[b].tolist()[0])
                lags_values_str = str(lags[b].tolist())
                prompt_ = (
                    f"<|start_prompt|>Dataset description: {self.description}"
                    f"Task description: impute the missing values; "
                    "Input statistics: "
                    f"min value {min_values_str}, "
                    f"max value {max_values_str}, "
                    f"median value {median_values_str}, "
                    f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                    f"top {self.top_k} lags are : {lags_values_str}<|end_prompt|>"
                )
                # prompt_ = (
                #     f"<|start_prompt|>Dataset description: {self.description}"
                #     "Task description: "
                #     f"given the observed information, "
                #     f"impute the missing values that indicated as 0; "  # in f{missing_mask[b].flatten()}
                #     "Input statistics: "
                #     f"min value {min_values_str}, "
                #     f"max value {max_values_str}, "
                #     f"median value {median_values_str}, "
                #     f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                #     f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
                # )

                prompt.append(prompt_)
            # x_enc [B * N, T, 1]
            x_enc = x_enc.reshape(B, N, T)  # [B, T, N]

            prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        if self.use_prompt:
            llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        else:
            llama_enc_out = enc_out
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out

    def interval_forecast(self, x_enc):
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        if self.use_prompt:
            x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

            min_values = torch.min(x_enc, dim=1)[0]
            max_values = torch.max(x_enc, dim=1)[0]
            medians = torch.median(x_enc, dim=1).values
            mean = torch.mean(x_enc, dim=1)  # shape: [64, 10]
            std = torch.std(x_enc, dim=1)
            lags = self.calcute_lags(x_enc)
            trends = x_enc.diff(dim=1).sum(dim=1)

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
            # x_enc [B * N, T, 1]
            x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()  # [B, T, N]

            prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        if self.use_prompt:
            llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        else:
            llama_enc_out = enc_out

        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        # mu = self.likelihood_layer_mu(dec_out[:, :, :, -self.patch_nums:])
        # sigma = torch.log(1 + torch.exp(self.likelihood_layer_sigma(dec_out[:, :, :, -self.patch_nums:]))) + 1e-6
        # mu = self.normalize_layers(mu.permute(0, 2, 1).contiguous(), 'denorm')
        # sigma = sigma.permute(0, 2, 1).contiguous() * self.normalize_layers.stdev
        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape  # [80, 18, 32]
        S, _ = source_embedding.shape  # [1000, 768]
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)  # [80, 18, 8, 128]
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)  # [1000, 8, 128]
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)  # [1000, 8, 128]

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)  # [80, 18, 8, 128]

        out = out.reshape(B, L, -1)  # [80, 18, 1024]

        return self.out_projection(out)  # [80, 18, 768]

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class Model(nn.Module):
    def __init__(self, configs,encoder_other_model='Transformer'):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.c_out = configs.c_out
        self.output_attention = configs.output_attention
        self.output_ori = configs.output_ori
        self.likelihood = configs.likelihood
        self.use_norm = configs.use_norm

        # Encoder
        self.encoder_other_model = encoder_other_model
        transformer_d_model = 64  # configs.d_model
        transformer_d_ff = transformer_d_model * 4  # configs.d_ff
        transformer_enc_layers = configs.e_layers  # configs.e_layers
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
                        transformer_d_model,  # configs.d_model,
                        transformer_d_ff,  # 4 * configs.d_model,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(transformer_enc_layers)
                ],
                norm_layer=torch.nn.LayerNorm(transformer_d_model)
            )
        self.LLM_encoder = LLMBlock(configs)
        if 'forecast' in self.task_name:
            self.encoder_linear_projection = nn.Linear(configs.seq_len, configs.pred_len)

        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'interval_forecast' or self.task_name == 'imputation' or self.task_name == 'imputation_forecast':
            # transformer_d_model = transformer_d_model
            self.dec_embedding = DataEmbedding(configs.c_out, transformer_d_model, configs.embed, configs.freq, configs.dropout)

            # Embedding
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
            self.projection = nn.Linear(transformer_d_model, configs.c_out)
            # self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
            # self.linear_predict = nn.Linear(configs.seq_len, configs.pred_len+configs.label_len)
        if self.task_name == 'imputation':
            self.output_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.output_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.output_projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

        self.use_forecast = configs.use_forecast
        if self.use_forecast:
            self.forecast_projection = nn.Linear(configs.forecast_dim, configs.enc_in)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        if self.use_forecast:
            means_forecast = x_forecast.mean(1, keepdim=True).detach()
            x_enc_forecast = x_forecast - means_forecast
            stdev_forecast = torch.sqrt(torch.var(x_enc_forecast, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_forecast /= stdev_forecast
            x_forecast_ = self.forecast_projection(x_forecast[:,-self.pred_len:,:])
            x_enc = torch.cat((x_enc, x_forecast_), dim=1)

        x_enc_target = x_enc[:,:,-self.c_out:]

        enc_out_target = self.LLM_encoder(x_enc_target)
        if self.encoder_other_model == 'Transformer':
            enc_in = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder_other(enc_in)
            # enc_out = self.encoder_linear_projection(enc_out.permute(0,2,1)).permute(0,2,1)
        elif self.encoder_other_model == 'LSTM':
            enc_out, (_) = self.encoder_other(x_enc)
            enc_out = self.encoder_linear_projection(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.encoder_other_model == 'Linear':
            enc_out = self.encoder_other(x_enc)
            enc_out = self.encoder_linear_projection(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        dec_in = enc_out_target
        dec_in = self.dec_embedding(dec_in, x_mark_dec[:,-self.label_len-self.pred_len:,:])

        dec_out = self.decoder(dec_in, enc_out, x_mask=None, cross_mask=None)
        dec_out = self.projection(dec_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, :1, -self.c_out:].repeat(1, self.pred_len+self.label_len, 1))
        dec_out = dec_out + (means[:, :1, -self.c_out:].repeat(1, self.pred_len+self.label_len, 1))

        return dec_out

    def interval_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        if self.use_forecast:
            means_forecast = x_forecast.mean(1, keepdim=True).detach()
            x_enc_forecast = x_forecast - means_forecast
            stdev_forecast = torch.sqrt(torch.var(x_enc_forecast, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_forecast /= stdev_forecast
            x_forecast_ = self.forecast_projection(x_forecast[:,-self.pred_len:,:])
            x_enc = torch.cat((x_enc, x_forecast_), dim=1)

        x_enc_target = x_enc[:,:,-self.c_out:]

        # 1
        enc_out_target = self.LLM_encoder(x_enc_target)
        if self.encoder_other_model == 'Transformer':
            enc_in = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder_other(enc_in)
            # enc_out = self.encoder_linear_projection(enc_out.permute(0,2,1)).permute(0,2,1)
        elif self.encoder_other_model == 'LSTM':
            enc_out, (_) = self.encoder_other(x_enc)
            enc_out = self.encoder_linear_projection(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.encoder_other_model == 'Linear':
            enc_out = self.encoder_other(x_enc)
            enc_out = self.encoder_linear_projection(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        dec_in = enc_out_target
        dec_in = self.dec_embedding(dec_in, x_mark_dec)
        dec_out = self.decoder(dec_in, enc_out, x_mask=None, cross_mask=None)
        dec_out, mu, sigma = self.likelihood_layer(dec_out)

        # 2
        # mu, sigma = self.LLM_encoder(x_enc_target, x_mark_enc, x_dec, x_mark_dec, x_forecast)
        # enc_out = self.encoder_other(x_enc)
        # dec_in = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out = self.decoder(dec_in, enc_out, x_mask=None, cross_mask=None)
        # dec_out = self.projection(dec_out)
        output_len = self.pred_len + self.seq_len
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer

            mu = mu * (stdev[:, :1, -self.c_out:].repeat(1, output_len, 1))
            mu = mu + (means[:, :1, -self.c_out:].repeat(1, output_len, 1))
            sigma = sigma * (stdev[:, :1, -self.c_out:].repeat(1, output_len, 1))
            dec_out = dec_out * (stdev[:, :1, -self.c_out:].repeat(1, self.pred_len+self.seq_len, 1))
            dec_out = dec_out + (means[:, :1, -self.c_out:].repeat(1, self.pred_len+self.seq_len, 1))

        return dec_out, mu, sigma

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        if self.use_norm:
            x_enc, means, stdev = masked_standardize_3d(x_enc, mask)

        x_enc_target = x_enc[:, :, -self.c_out:]
        enc_out_target = self.LLM_encoder(x_enc_target,mask[:, :, -self.c_out:])

        if self.encoder_other_model == 'Transformer':
            enc_in = self.enc_embedding(x_enc[:, :, :-self.c_out], x_mark_enc)
            enc_out, attns = self.encoder_other(enc_in, attn_mask=mask[:, :, :-self.c_out])
        elif self.encoder_other_model == 'LSTM':
            enc_out, (_) = self.encoder_other(x_enc)
        elif self.encoder_other_model == 'Linear':
            enc_out = self.encoder_other(x_enc)  # x_enc[:, :, :-self.c_out]
        enc_out_target = mask[:, :, -self.c_out:] * x_enc[:, :, -self.c_out:] + (1 - mask[:, :, -self.c_out:]) * enc_out_target

        dec_in = self.dec_embedding(enc_out_target, x_mark_enc)
        dec_out = self.decoder(dec_in, enc_out, x_mask=None, cross_mask=mask)
        dec_out = self.projection(dec_out)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            enc_out_target = enc_out_target * (stdev[:, :1, -self.c_out:].repeat(1, self.pred_len + self.seq_len, 1))
            enc_out_target = enc_out_target + (means[:, :1, -self.c_out:].repeat(1, self.pred_len + self.seq_len, 1))
            dec_out = dec_out * (stdev[:, :1, -self.c_out:].repeat(1, self.pred_len + self.seq_len, 1))
            dec_out = dec_out + (means[:, :1, -self.c_out:].repeat(1, self.pred_len + self.seq_len, 1))
        return enc_out_target, dec_out

    def imputation_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        if self.use_norm:

            # Normalization from Non-stationary Transformer
            means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
            means = means.unsqueeze(1).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(1).detach()
            x_enc /= stdev

        x_enc_target = x_enc[:, :, -self.c_out:]
        enc_out_target = self.LLM_encoder(x_enc_target)

        if self.encoder_other_model == 'Transformer':
            enc_in = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder_other(enc_in, attn_mask=mask)
        elif self.encoder_other_model == 'LSTM':
            enc_out, (_) = self.encoder_other(x_enc)
        elif self.encoder_other_model == 'Linear':
            enc_out = self.encoder_other(x_enc)
        enc_out = self.encoder_linear_projection(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        enc_out_target[:,:self.seq_len,:] = mask[:, :, -self.c_out:] * x_enc_target[:,:self.seq_len,:] + (1 - mask[:, :, -self.c_out:]) * enc_out_target[:,:self.seq_len,:]
        dec_in = self.dec_embedding(enc_out_target, x_mark_dec)

        dec_out = self.decoder(dec_in, enc_out, x_mask=None, cross_mask=mask)
        dec_out = self.projection(dec_out)
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, :1, -self.c_out:].repeat(1, self.pred_len + self.seq_len, 1))
            dec_out = dec_out + (means[:, :1, -self.c_out:].repeat(1, self.pred_len + self.seq_len, 1))
        return enc_out_target[:,:self.seq_len,:], dec_out

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
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,x_forecast)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out1, dec_out2 = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            if self.output_ori:
                dec_out2 = mask[:, :, -self.c_out:] * x_enc[:, :, -self.c_out:] + (1 - mask[:, :, -self.c_out:]) * dec_out2
            return dec_out2   # [B, L, D]
        if self.task_name == 'imputation_forecast':
            enc_out_target, dec_out = self.imputation_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            if self.output_ori:
                dec_out[:, :self.seq_len, -self.c_out:] = mask[:, :, -self.c_out:] * x_enc[:, :self.seq_len, -self.c_out:] + (1 - mask[:, :, -self.c_out:]) * dec_out[:, :self.seq_len, -self.c_out:]
            return enc_out_target, dec_out   # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == 'interval_forecast':
            dec_out, mu, sigma = self.interval_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast)
            return dec_out[:, -self.pred_len:, :], mu[:, -self.pred_len:, :], sigma[:, -self.pred_len:, :]
        return None