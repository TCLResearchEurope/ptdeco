[![license](https://img.shields.io/github/license/TCLResearchEurope/ptdeco)](https://opensource.org/license/apache-2-0/)
[![check](https://github.com/TCLResearchEurope/ptdeco/actions/workflows/check.yml/badge.svg)](https://github.com/TCLResearchEurope/ptdeco/actions/workflows/check.yml)
[![test](https://github.com/TCLResearchEurope/ptdeco/actions/workflows/test.yml/badge.svg)](https://github.com/TCLResearchEurope/ptdeco/actions/workflows/test.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
[![Type checker: mypy](https://img.shields.io/badge/type_checker-mypy-black)](https://github.com/python/mypy)

# ptdeco

`ptdeco` is a library for model optimization by matrix decomposition built on top of PyTorch.

There is an introductory presentation about `ptdeco` from Warsaw AI meetup
2024.05.23 - [Practical low-rank decomposition (not only) for large language
models](https://www.youtube.com/watch?v=8CcRsX4IMnU&t=1800s).

Contents of this README:

* [Introduction](#introduction)
* [Installation](#installation)
* [Saving and loading a decomposed model](#saving-and-loading-a-decomposed-model)
   * [Saving a decomposed model](#saving-a-decomposed-model)
   * [Loading a decomposed model](#loading-a-decomposed-model)
* [Links to other methods for model compression by decomposition](#links-to-other-methods-for-model-compression-by-decomposition)

## Introduction

Currently, `ptdeco` implements the following methods:

* **dwain** - iterative method based on low-rank decomposition of features
  (dwain =  **D**ecomposing **W**eights **A**lgorithm - an **I**terative tech**N**ique). Tested on **LLMs** (large language models) and **vision models**

* **lockd** - method based on local knowledge distillation.
  (lockd = **LOC**al **K**nowledge **D**istillation). Tested on **vision models**

* **falor** - method based on low-rank decomposition of features inspired by [Compressing Transformers: Features Are Low-Rank, but Weights Are Not! by Yu Hao, Wu Jianxin (2023)](https://doi.org/10.1609/aaai.v37i9.26304), (falor = **F**eatures **Are** **LO**w **R**ank). Tested on **vision models**


**dwain** method does not require pretraining. It can decompose linear layers and
1x1 convolutions.

**lockd** method requires short (~ 10 ImageNet epochs) knowledge distillation
pretraining before decomposition is made. It can decompose linear layers and
convolutions.

**falor** method does not require pretraining. Model decomposition lasts < 1
GPU hour (depending on model size and parameters). It can decompose linear
layers and 1x1 convolutions.


## Installation

```bash
pip install ptdeco
```
## Saving and loading a decomposed model

### Saving a decomposed model

As a result of decomposition you get `decompose_config` dictionary. You need to
serialize this e.g. to JSON. This will let you recreate the structure of a
decomposed model.  Except this, you need to save `state_dict` to recover
the weights of a decomposed model. The code below illustrates the procedure:

```python
import json
import pathlib

# Your decomposition code

output_path = pathlib.Path("YOUR/CHEKCPOINT/DIRECTORY")
out_decompose_config_path = output_path / "decompose_config.json"
with open(out_decompose_config_path, "wt") as f:
    json.dump(decompose_config, f)
out_decompose_state_dict_path = output_path / "decompose_state_dict.pt"
torch.save(model.state_dict(), out_decompose_state_dict_path)
```

### Loading a decomposed model
To load the model, you need to recreate the original model first. Next, you load and apply the
`decompose_config`. Finally, you load the `state_dict` (note the state dict "fits" the
decomposed model, so you need to do it as a last step). The code below illustrates
the procedure:

```python

import json
import pathlib

import ptdeco

model = ... # Build original model
device = ...     # Specify the device original model uses

output_path = pathlib.Path("YOUR/CHEKCPOINT/DIRECTORY")

with open(output_path / "decompose_config.json", "rt") as f:
        decompose_config = json.load(f)

ptdeco.utils.apply_decompose_config_in_place(model, decompose_config)

sd = torch.load(output_path / "decompose_state_dict.pt")

model.load_state_dict(sd, map_location=device)

# Now `model` is decomposed and contains appropriate weights
```

## Links to other methods for model compression by decomposition

Other methods using decomposition for model compression, not implemented in this package:

+ [(2024) Feature-based Low-Rank Compression of Large Language Models via Bayesian Optimization by Ji Yixin, Xiang Yang, Li Juntao, Chen Wei, Liu Zhongyi, Chen Kehai, Zhang Min](https://arxiv.org/pdf/2405.10616)

+ [(2024) SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression by Wang Xin, Zheng Yu, Wan Zhongwei, Zhang Mi](https://arxiv.org/pdf/2403.07378)

+ [(2024) SliceGPT: Compress Large Language Models by Deleting Rows and Columns by Ashkboos Saleh, Croci Maximilian L., Nascimento Marcelo Gennari do, Hoefler Torsten, Hensman James](https://arxiv.org/pdf/2401.15024)

+ [(2023) ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models by Yuan Zhihang, Shang Yuzhang, Song Yue, Wu Qiang, Yan Yan, Sun Guangyu](https://arxiv.org/pdf/2312.05821)

+ [(2023) LORD: Low Rank Decomposition Of Monolingual Code LLMs For One-Shot Compression by Kaushal Ayush, Vaidhya Tejas, Rish Irina](https://arxiv.org/pdf/2309.14021)

+ [(2023) LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation by Li Yixiao, Yu Yifan, Zhang Qingru, Liang Chen, He Pengcheng, Chen Weizhu, Zhao Tuo](https://arxiv.org/pdf/2306.11222)

+ [(2023) Rethinking Compression: Reduced Order Modelling of Latent Features in Large Language Models by Chavan Arnav, Lele Nahush, Gupta Deepak](https://arxiv.org/pdf/2312.07046)

+ [(2023) The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction by Sharma Pratyusha, Ash Jordan T., Misra Dipendra](https://arxiv.org/pdf/2312.13558)

+ [(2022) Numerical Optimizations for Weighted Low-rank Estimation on Language Model by Hua Ting, Hsu Yen-Chang, Wang Felicity, Lou Qian, Shen Yilin, Jin Hongxia](https://arxiv.org/pdf/2211.09718)

+ [(2022) Language model compression with weighted low-rank factorization by Hsu Yen-Chang, Hua Ting, Chang Sungen, Lou Qian, Shen Yilin, Jin Hongxia](https://arxiv.org/pdf/2207.00112)

+ [(2021) DRONE: Data-aware Low-rank Compression for Large NLP Models by Chen Patrick H., Yu Hsiang-Fu, Dhillon I., Hsieh Cho-Jui](https://proceedings.neurips.cc/paper/2021/file/f56de5ef149cf0aedcc8f4797031e229-Paper.pdf)

+ [(2020) Compressing Pre-trained Language Models by Matrix Decomposition by Noach Matan Ben, Goldberg Yoav](https://aclanthology.org/2020.aacl-main.88.pdf)
