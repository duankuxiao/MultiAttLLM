# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based research framework for **time series imputation** using Large Language Models (LLMs). The project implements **ImputeLLM**, a model that leverages pre-trained LLMs (GPT-2, BERT, LLAMA, Qwen) for missing value imputation in multivariate time series.

## Common Commands

```bash
# Run imputation experiments (trains multiple models across mask rates 0.1-0.5)
python run_imputation.py

# Models tested: RNN, DLinear, Transformer, Informer, iTransformer, PatchTST, TimesNet, LLMformer
```

## Architecture

### Data Flow
```
Input -> Normalization -> [LLM Encoder (targets) + Transformer Encoder (covariates)]
      -> Cross-Attention Decoder -> Output Projection -> Denormalization -> Imputed Output
```

### Key Components
- **ImputeLLM** (`models/ImputeLLM.py`): Main model with:
  - `LLMBlock`: Frozen LLM with multi-head attention reprogramming
  - `Model`: Dual-encoder (LLM for targets, Transformer for covariates) + decoder
  - Dynamic prompts with statistical context (min, max, median, trend, lags)

- **Experiment Runner** (`exp/exp_imputation.py`): Handles train/val/test with adaptive loss weighting

### Directory Structure
- `models/`: Neural network implementations (ImputeLLM + baselines)
- `layers/`: Reusable components (Embed, Attention, Encoder/Decoder)
- `exp/`: Experiment runners (exp_basic.py, exp_imputation.py)
- `configs/`: Configuration (common_configs.py has 137 CLI args, HVAC_configs.py for dataset)
- `data_provider/`: Dataset loading and preprocessing
- `utils/`: Metrics, masking methods, loss functions

### Model Registry
Models registered in `exp/exp_basic.py:model_dict`:
- LLM-enhanced: ImputeLLM, TimeLLM
- Baselines: TimesNet, DLinear, Informer, Transformer, iTransformer, RNN, PatchTST

## Configuration

### Key Parameters (`configs/common_configs.py`)
| Parameter | Description |
|-----------|-------------|
| `seq_len` | Input sequence length (default: 48) |
| `mask_rate` | Missing value ratio (0.1-0.5) |
| `mask_method` | mcar/mar/rdo |
| `llm_model` | GPT2/BERT/LLAMA1b/LLAMA3b/LLAMA8b/QWEN |
| `llm_layers` | Number of LLM layers to use |
| `use_prompt` | Enable dynamic statistical prompts |
| `loss_method` | fix/adaptive (hybrid loss weighting) |

### Hyperparameter Setup
Model-specific hyperparameters in `setup_ImputeLLM.py` - automatically configures d_model, d_ff, e_layers based on model type.

### LLM Paths
Models expect pre-trained LLMs at `D:\LLM\`:
- GPT2, BERT (768-dim)
- LLAMA1b (2048-dim), LLAMA3b (3072-dim), LLAMA8b (4096-dim)
- Qwen

## Dataset
HVAC dataset (`configs/HVAC_configs.py`):
- 25 input features, 4 target variables
- Targets: Total_Power, Total_Chiller_Power, System_Energy_Efficiency, Total_Cooling_Capacity
- 15-minute intervals (2023-2024)

## Output
Results saved to `results/` directory:
- `{model_id}_{mask_method}_benchmark_metrics.csv`: Overall metrics
- `{model_id}_{mask_method}_benchmark_imputation_metrics.csv`: Imputation-specific metrics
