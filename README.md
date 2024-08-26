[![license](https://img.shields.io/github/license/TCLResearchEurope/ptdeco)](https://opensource.org/license/apache-2-0/)
[![check](https://github.com/TCLResearchEurope/ptdeco/actions/workflows/check.yml/badge.svg)](https://github.com/TCLResearchEurope/ptdeco/actions/workflows/check.yml)
[![test](https://github.com/TCLResearchEurope/ptdeco/actions/workflows/test.yml/badge.svg)](https://github.com/TCLResearchEurope/ptdeco/actions/workflows/test.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
[![Type checker: mypy](https://img.shields.io/badge/type_checker-mypy-black)](https://github.com/python/mypy)

# ptdeco

`ptdeco` is a library for model optimization by matrix decomposition built on top of PyTorch.

There is a introductory presentation on `ptdeco` from Warsaw AI meetup 2024.05.23 - [Practical low-rank decomposition (not only) for large language models](https://www.youtube.com/watch?v=8CcRsX4IMnU&t=1800s).

<details>

<summary>Table of contents</summary>

* [Introduction](#introduction)
* [Installation](#installation)
* [Saving and loading a decomposed model](#saving-and-loading-a-decomposed-model)
   * [Saving a decomposed model](#saving-a-decomposed-model)
   * [Loading a decomposed model](#loading-a-decomposed-model)

</details>

## Introduction

Currently, `ptdeco` implements the following methods:

* **lockd** - method based on local knowledge distillation, tested on vision models
  (lockd = **LOC**al **K**nowledge **D**istillation)

* **falor** - method based on low-rank decomposition of features inspired by [Compressing Transformers: Features Are Low-Rank, but Weights Are Not! by Yu Hao, Wu Jianxin (2023)](https://doi.org/10.1609/aaai.v37i9.26304), tested on vision models
  (falor = **F**eatures **Are** **LO**w **R**ank)

* **dwain** - iterative method based on low-rank decomposition of features, tested on Large Language Models
  (dwain =  **D**ecomposing **W**eights **A**lgorithm - an **I**terative tech**N**ique)

**lockd** method requires short (~ 10 ImageNet epochs) knowledge distillation
pretraining before decomposition is made. It can decompose linear layers and
convolutions.

**falor** method does not require pretraining. Model decomposition lasts < 1
GPU hour (depending on model size and parameters). It can decompose linear
layers and 1x1 convolutions.

**dwain** method does not require pretraining. It can decompose linear layers and
1x1 convolutions.

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
