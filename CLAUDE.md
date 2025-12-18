# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based research framework for **long-term time series forecasting** using Large Language Models (LLMs). The project implements **MultiAttLLM**, which combines multi-head cross-attention reprogramming with pre-trained LLMs for electricity demand and renewable energy forecasting.

**Paper**: "A novel attention-enhanced LLM approach for accurate power demand and generation forecasting" (Renewable Energy, 2025)

## Common Commands

```bash
# Run forecasting benchmark (trains multiple models)
python run_long_term_forecast.py

# Zero-shot transfer testing across regions
python zero-shot_test.py

# Models benchmarked: RNN, DLinear, Informer, Autoformer, iTransformer, TimesNet, PatchTST, TimeLLM, MultiAttLLM
```

## Architecture

### Data Flow
```
Input -> [LLM Encoder (targets) | Transformer Encoder (covariates)]
      -> Cross-Attention Reprogramming -> Self-Attention Fusion -> Output Projection -> Predictions
```

### Key Components
- **MultiAttLLM** (`models/MultiAttLLM.py`): Main model with:
  - `LLMBlock`: Frozen LLM with cross-attention reprogramming (converts time series patches to LLM token space)
  - `CrossAttentionLayer`: Aligns time series patches with LLM word embeddings
  - `Model`: Dual-encoder (LLM for targets, Transformer for covariates) + decoder fusion

- **Experiment Runner** (`exp/exp_forecasting.py`): Handles train/val/test loops

### Directory Structure
- `models/`: Neural network implementations (MultiAttLLM + baselines)
- `layers/`: Reusable components (Embed, Attention, Encoder/Decoder)
- `exp/`: Experiment runners (exp_basic.py, exp_forecasting.py)
- `configs/`: Configuration (common_configs.py, electricity_configs.py)
- `data_provider/`: Dataset loading (data_loader_LLM.py for LLM models)
- `utils/`: Metrics, tools, loss functions

### Model Registry
Models registered in `exp/exp_basic.py:model_dict`:
- LLM-enhanced: MultiAttLLM, TimeLLM
- Baselines: TimesNet, DLinear, Informer, Transformer, iTransformer, RNN, PatchTST

## Configuration

### Key Parameters (`configs/electricity_configs.py`)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `seq_len` | Input sequence length | 72 |
| `pred_len` | Prediction horizon | 168 (7 days) |
| `label_len` | Decoder prefix length | 72 |
| `llm_model` | GPT2/BERT/LLAMA/QWEN | GPT2 |
| `llm_layers` | Number of LLM layers | 6 |
| `c_out` | Number of target variables | 1 |

### Hyperparameter Setup
Model-specific hyperparameters in `setup_MultiAttLLM.py` - automatically configures d_model, d_ff, e_layers, learning_rate based on model type.

### LLM Paths
Models expect pre-trained LLMs at `D:\LLM\`:
- GPT2, BERT (768-dim)
- LLAMA (2048-dim), LLAMA3b (3072-dim), LLAMA8b (4096-dim)

## Dataset

Electricity demand datasets (`dataset/electricity/`):
- Regions: tokyo, kyushu, hokkaido, tohoku
- 21 features including weather, solar, wind data
- Targets: Electricity, Renewable_energy, Coal
- Hourly intervals, 2+ years of data

## Output

Results saved to `results/` directory:
- Model comparison CSV files
- Zero-shot transfer results across regions
