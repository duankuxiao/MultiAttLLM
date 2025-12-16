# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based research framework for time series **forecasting** and **imputation** using Large Language Models (LLMs) combined with Transformer architectures. The project implements methods from several academic papers.

### Key Models
- **MultiAttLLM** (`models/MultiAttLLM.py`): LLM-enhanced forecasting model with cross-attention reprogramming
- **LLMformer** (`models/LLMformer.py`): Hybrid Transformer+LLM for imputation with probabilistic outputs
- **ImputeLLM** (`models/ImputeLLM.py`): Dedicated imputation model with extended LLM support

## Common Commands

### Training
```bash
# Run imputation experiments (tests multiple models and mask rates)
python run_imputation.py

# Run forecasting experiments
python run_main.py

# Zero-shot testing
python zero-shot_test.py
```

### Dependencies
- PyTorch with CUDA support
- Hugging Face Transformers
- NumPy, Pandas, Scikit-learn, Matplotlib

### LLM Model Paths
Models expect pre-trained LLMs at `D:\LLM\`:
- `gpt2`, `bert`, `qwen` (768-dim)
- `llama`, `llama1b` (2048-dim), `llama3b` (3072-dim), `llama8b` (4096-dim)

## Architecture

### Data Flow
```
Input → Normalization → [LLM Encoder (target) + Transformer Encoder (covariates)]
      → Cross-Attention/Fusion → Decoder → Output Projection → Denormalization → Output
```

### Key Directories
- `models/`: Neural network model implementations (12 models including baselines)
- `layers/`: Reusable components (Embed, Attention, Encoder/Decoder blocks)
- `exp/`: Experiment runners (`exp_imputation.py`, `exp_forecasting.py`)
- `configs/`: Configuration definitions and hyperparameters
- `data_provider/`: Dataset loading and preprocessing
- `utils/`: Metrics, masking, loss functions, tools

### Configuration System
- `configs/common_configs.py`: 137 command-line arguments for all settings
- `configs/HVAC_configs.py`: HVAC dataset configuration (25 features, 4 targets)
- `configs/electricity_configs.py`: Electricity forecasting configuration
- `hyparam_imputation.py` / `hyparam_setup_forele.py`: Model-specific hyperparameters

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
Models are registered in `exp/exp_basic.py:model_dict`:
- TimesNet, DLinear, Informer, Transformer, iTransformer, PatchTST, RNN
- LLMformer, MultiAttLLM, ImputeLLM, TimeLLM

### Datasets
- `dataset/electricity/`: Regional electricity data (kyushu, hokkaido, tohoku, tokyo)
- `dataset/HVAC/summary_2.csv`: HVAC system data (2023-2024, 15-min intervals)
- `dataset/prompt_bank/`: Domain descriptions for LLM prompt engineering

### Output
Results saved to `results/` directory as CSV files with metrics (MAE, MSE, RMSE).
