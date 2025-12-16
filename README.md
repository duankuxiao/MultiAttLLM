# MultiAttLLM: Multi-Attention LLM for Energy Demand Forecasting

This repository contains the official implementation of the paper **"MultiAttLLM: Multi-Attention Large Language Model for Energy Demand and Generation Forecasting"**.

## Overview

MultiAttLLM is a novel architecture that combines Large Language Models (LLMs) with multi-head cross-attention mechanisms for time series forecasting. The model leverages pre-trained LLM knowledge through a reprogramming approach to improve long-term electricity demand and renewable energy generation forecasting.

## Key Features

- **LLM Integration**: Supports GPT-2, BERT, LLAMA (1b/3b/8b), and Qwen models
- **Cross-Attention Reprogramming**: Aligns time series patches with LLM token embeddings
- **Multi-variate Forecasting**: Handles multiple target variables with covariate features
- **Long-horizon Prediction**: Designed for 7-day (168-hour) forecasting horizons
![Framework of MultiAttLLM](figures/MultiAttLLM/workflow.png)
## Architecture

```
Input → Normalization → Cross Attention → [LLM Encoder (target) + Transformer Encoder (covariates)]
      → self-Attention Fusion → Output Projection → Predictions
```
![Framework of MultiAttLLM](figures/MultiAttLLM/MultiAttLLM.png)

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers (Hugging Face)
- NumPy, Pandas, Scikit-learn

## Installation

```bash
pip install torch transformers numpy pandas scikit-learn matplotlib
```

## Usage

### Training

```bash
python run_main.py
```

### Configuration

Key parameters in `configs/electricity_configs.py`:
- `seq_len`: Input sequence length (default: 72)
- `pred_len`: Prediction horizon (default: 168)
- `llm_model`: LLM backbone (GPT2/BERT/LLAMA/QWEN)
- `llm_layers`: Number of LLM layers to use

### Hyperparameters

Model-specific hyperparameters are in `hyparam_setup_forele.py`.

## Dataset

The model is evaluated on electricity demand datasets from multiple regions:
- Tokyo, Kyushu, Hokkaido, Tohoku (Japan)
![Framework of MultiAttLLM](figures/MultiAttLLM/datasets.png)
- 
Data should be placed in `dataset/electricity/` directory.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{multiattllm2025,
  title={MultiAttLLM: Multi-Attention Large Language Model for Energy Demand and Generation Forecasting},
  author={},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License.

## Related Papers

- [LLMformer (Probabilistic Forecasting)](../paper/LLMformer)
- [ImputeLLM (Time Series Imputation)](../paper/ImputeLLM)
