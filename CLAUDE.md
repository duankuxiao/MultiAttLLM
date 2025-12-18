# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based research framework for time series **forecasting** and **imputation** using Large Language Models (LLMs) combined with Transformer architectures. The project implements methods from three academic papers, each on a separate branch.

## Common Commands

```bash
# Run probabilistic forecasting experiments
python run_probabilistic_forecast.py

```

## Architecture

### Data Flow
```
Input -> Normalization -> [LLM Encoder (target) + Transformer Encoder (covariates)]
      -> Cross-Attention/Fusion -> Decoder -> Output Projection -> Denormalization -> Output
```

### Directory Structure
- `models/`: Neural network implementations (MultiAttLLM, LLMformer, ImputeLLM, plus baselines)
- `layers/`: Reusable components (Embed, Attention, Encoder/Decoder blocks)
- `exp/`: Experiment runners (`exp_imputation.py`, `exp_forecasting.py`)
- `configs/`: Configuration definitions and hyperparameters
- `data_provider/`: Dataset loading and preprocessing
- `utils/`: Metrics, masking, loss functions

### LLM Backend Configuration
Models expect pre-trained LLMs at `D:\LLM\`:
- GPT2, BERT, Qwen (768-dim)
- LLAMA 1b (2048-dim), 3b (3072-dim), 8b (4096-dim)

### Key Configuration Parameters
| Parameter | Description |
|-----------|-------------|
| `seq_len` | Input sequence length |
| `pred_len` | Prediction horizon (forecasting) |
| `label_len` | Decoder prefix length |
| `c_out` | Number of target variables |
| `mask_rate` | Missing value ratio (imputation) |
| `mask_method` | mcar/mar/rdo |
| `llm_model` | GPT2/LLAMA/BERT/QWEN |
| `llm_layers` | Number of LLM layers to use |

### Model Registry
Models are registered in `exp/exp_basic.py:model_dict`. Available models include:
- LLM-enhanced: LLMformer
- Baselines: TimesNet, DLinear, Informer, Transformer, iTransformer, PatchTST, RNN

## Requirements
- Python 3.8+
- PyTorch 1.12+ with CUDA
- Hugging Face Transformers
- NumPy, Pandas, Scikit-learn, Matplotlib
