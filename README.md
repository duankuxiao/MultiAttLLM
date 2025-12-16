# LLM-Enhanced Time Series Models

This repository contains implementations for three research papers on Large Language Model (LLM) enhanced time series analysis.

## Papers and Branches

| Branch | Paper                                                            | Task | Model |
|--------|------------------------------------------------------------------|------|-------|
| [`paper/MultiAttLLM`](../../tree/paper/MultiAttLLM) | Multi-Attention LLM for Energy Demand and Generation Forecasting | Long-term Forecasting | MultiAttLLM |
| [`paper/LLMformer`](../../tree/paper/LLMformer) | Hybrid LLM-Transformer for Probabilistic Forecasting             | Probabilistic Forecasting | LLMformer |
| [`paper/ImputeLLM`](../../tree/paper/ImputeLLM) | LLM Enhanced Time Series Imputation                              | Missing Value Imputation | ImputeLLM |

## Quick Start

Clone a specific branch for the paper you're interested in:

```bash
# For energy demand forecasting
git clone -b paper/MultiAttLLM https://github.com/YOUR_USERNAME/MultiAttLLM.git

# For probabilistic forecasting
git clone -b paper/LLMformer https://github.com/YOUR_USERNAME/MultiAttLLM.git

# For time series imputation
git clone -b paper/ImputeLLM https://github.com/YOUR_USERNAME/MultiAttLLM.git
```

## Overview

### MultiAttLLM (Forecasting)
- **Task**: Long-term electricity demand and renewable energy forecasting
- **Horizon**: 7-day (168-hour) predictions
- **Key Feature**: Multi-head cross-attention reprogramming between time series and LLM embeddings

### LLMformer (Probabilistic Forecasting)
- **Task**: Probabilistic time series forecasting and imputation
- **Output**: Distribution parameters (mean, standard deviation) for uncertainty quantification
- **Key Feature**: Hybrid dual-encoder architecture with likelihood-based output

### ImputeLLM (Imputation)
- **Task**: Missing value imputation in time series data
- **Patterns**: MCAR, MAR, RDO missing data patterns
- **Key Feature**: Dynamic prompts with statistical context for LLM guidance

## Supported LLM Backends

All models support multiple pre-trained LLMs:
- GPT-2 (768-dim)
- BERT (768-dim)
- LLAMA 1b/3b/8b (2048/3072/4096-dim)
- Qwen (varies)

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers (Hugging Face)
- NumPy, Pandas, Scikit-learn

## Citation

If you use this code, please cite the relevant paper(s):

```bibtex
@article{hu2025novel,
  title={A novel attention-enhanced LLM approach for accurate power demand and generation forecasting},
  author={Hu, Zehuan and Gao, Yuan and Sun, Luning and Mae, Masayuki},
  journal={Renewable Energy},
  pages={123465},
  year={2025},
  publisher={Elsevier}
}

@article{llmformer2025,
  title={LLMformer: Hybrid LLM-Transformer Model for Probabilistic Time Series Forecasting},
  year={2025}
}

@article{imputellm2025,
  title={ImputeLLM: Large Language Model Enhanced Time Series Imputation},
  year={2025}
}
```

## License

This project is licensed under the MIT License.
