[![license](https://img.shields.io/github/license/TCLResearchEurope/ptdeco)](https://opensource.org/license/apache-2-0/)
[![check](https://github.com/TCLResearchEurope/ptdeco/actions/workflows/check.yml/badge.svg)](https://github.com/TCLResearchEurope/ptdeco/actions/workflows/check.yml)
[![test](https://github.com/TCLResearchEurope/ptdeco/actions/workflows/test.yml/badge.svg)](https://github.com/TCLResearchEurope/ptdeco/actions/workflows/test.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

# ptdeco

`ptdeco` is a library for model optimization by decomposition built on top of PyTorch.

<details>

<summary>Table of contents</summary>

* [Introduction](#introduction)
* [Sample results](#sample-results)
  * [convnext_femto.d1_in1k](#convnext_femtod1_in1k)
  * [convnextv2_nano.fcmae_ft_in22k_in1k](#convnextv2_nanofcmae_ft_in22k_in1k)
  * [rexnetr_200.sw_in12k_ft_in1k](#rexnetr_200sw_in12k_ft_in1k)
  * [efficientformerv2_s2.snap_dist_in1k](#efficientformerv2_s2snap_dist_in1k)
  * [mobilevitv2_200.cvnets_in22k_ft_in1k](#mobilevitv2_200cvnets_in22k_ft_in1k)
  * [swinv2_cr_tiny_ns_224.sw_in1k](#swinv2_cr_tiny_ns_224sw_in1k)
  * [deit3_small_patch16_224.fb_in1k](#deit3_small_patch16_224fb_in1k)
  * [resnet18.a2_in1k](#resnet18a2_in1k)
  * [resnet50d.a1_in1k](#resnet50da1_in1k)

</details>

## Introduction

Currently, `ptdeco` implements two methods:

* **lockd** - our custom method based on local knowledge distillation
  (lockd = **LOC**al **K**nowledge **D**istillation)

* **falor** - described in [Compressing Transformers: Features Are Low-Rank, but Weights Are Not! by Yu Hao, Wu Jianxin (2023)](https://doi.org/10.1609/aaai.v37i9.26304)
  (falor = **F**eatures **Are** **LO**w **R**ank)

**lockd** method requires short (~ 10 ImageNet epochs) knowledge distillation
pretraining before decomposition is made. It can decompose linear layers and
convolutions.

**falor** method does not require pretraining. Model decomposition lasts < 1
GPU hour (depending on model size and parameters). It can decompose linear
layers and 1x1 convolutions.


## Sample results

### convnext_femto.d1_in1k

Resolution: 224 x 224

| name        |   params |   kmapps |   acc |   epochs_ft | method   | settings           |
|-------------|----------|----------|-------|-------------|----------|--------------------|
| baseline    |     5.22 |    30.57 | 77.56 |             |          |                    |
| lockd_nt010 |     4.13 |    22.82 | 76.48 |         200 | lockd    | nsr_threshold=0.10 |

On device timing results:

* convnext_femto.d1_in1k - t1_pro_gpu

  | name             |   latency |   latency_std |   speedup [%] |
  |------------------|-----------|---------------|---------------|
  | baseline         |      81.6 |           1.5 |               |
  | lockd_nt010      |      67.3 |           1.6 |          17.5 |

* convnext_femto.d1_in1k - t1_pro_cpu1

  | name             |   latency |   latency_std |   speedup [%] |
  |------------------|-----------|---------------|---------------|
  | baseline         |     123.0 |           0.3 |               |
  | lockd_nt010      |     112.0 |           0.5 |           8.9 |

* convnext_femto.d1_in1k - t1_pro_cpu4

  | name             |   latency |   latency_std |   speedup [%] |
  |------------------|-----------|---------------|---------------|
  | baseline         |      68.6 |           4.1 |               |
  | decomposed_nt010 |      63.6 |           4.4 |           7.4 |


### convnextv2_nano.fcmae_ft_in22k_in1k

Resolution: 224 x 224

| name        |   params |   kmapps |   acc |   epochs_ft | method   | settings           |
|-------------|----------|----------|-------|-------------|----------|--------------------|
| baseline    |    15.62 |    95.59 | 81.97 |             |          |                    |
| lockd_nt005 |    13.33 |    78.86 | 82.09 |         200 | lockd    | nsr_threshold=0.05 |
| lockd_nt010 |    10.78 |    63.61 | 81.60 |         200 | lockd    | nsr_threshold=0.10 |


### rexnetr_200.sw_in12k_ft_in1k

Resolution: 224 x 224

| name        |   params |   kmapps |   acc |   epochs_ft | mehtod   | settings           |
|-------------|----------|----------|-------|-------------|----------|--------------------|
| baseline    |    16.52 |    61.88 | 82.41 |             |          |                    |
| lockd_nt010 |    13.59 |    45.81 | 81.78 |         200 | lockd    | nsr_threshold=0.10 |


### efficientformerv2_s2.snap_dist_in1k

Resolution: 224 x 224

| name        |   params |   kmapps |   acc |   epochs_ft | method   | settings           |
|-------------|----------|----------|-------|-------------|----------|--------------------|
| baseline    |    12.71 |    49.50 | 82.20 |             |          |                    |
| lockd_nt010 |    11.64 |    42.84 | 80.54 |         200 | lockd    | nsr_threshold=0.10 |
| lockd_nt015 |     9.74 |    35.45 | 75.37 |         200 | lockd    | nsr_threshold=0.15 |


### mobilevitv2_200.cvnets_in22k_ft_in1k

Resolution: 256 x 256

Remarks:
* normalization zero to one

| name        |   params |   kmapps |   acc |   epochs_ft | method   | settings           |
|-------------|----------|----------|-------|-------------|----------|--------------------|
| baseline    |    18.45 |   215.15 | 82.28 |             |          |                    |
| lockd_nt010 |    11.14 |    99.28 | 81.71 |         200 | lockd    | nsr_threshold=0.10 |


### swinv2_cr_tiny_ns_224.sw_in1k

Resolution: 224 x 224

| name         |   params |   kmapps |   acc |   epochs_ft | method   | settings                   |
|--------------|----------|----------|-------|-------------|----------|----------------------------|
| baseline     |    28.33 |   181.44 | 81.53 |             |          |                            |
| lockd_nt010  |    15.62 |   102.82 | 81.26 |         200 | lockd    | nsr_threshold=0.10         |
| falor_nft005 |    17.22 |   103.43 | 80.99 |         200 | falor    | nsr_final_threshold=0.0455 |


### deit3_small_patch16_224.fb_in1k

Resolution: 224 x 224


| name         |   params |   kmapps |   acc |   epochs_ft | method   | settings                  |
|--------------|----------|----------|-------|-------------|----------|---------------------------|
| baseline     |    22.06 |   165.46 | 81.30 |             |          |                           |
| lockd_nt010  |    14.53 |   107.75 | 81.30 |         200 | lockd    | nsr_threshold=0.10        |
| falor_nft005 |    14.51 |   107.56 | 81.31 |         200 | falor    | nsr_final_threshold=0.045 |


### resnet18.a2_in1k

Method: lockd

Resolution: 224 x 224

| name         |   nsr_thr |   params |   kmapps |   acc |
|--------------|-----------|----------|----------|-------|
| original     |           |    11.69 |       71 | 70.87 |
| decomposed_1 |      0.03 |     9.23 |       56 | 62.88 |
| decomposed_2 |      0.05 |     7.77 |       46 | 62.80 |
| decomposed_3 |      0.10 |     5.84 |       34 | 62.58 |


### resnet50d.a1_in1k

Method: lockd

Resolution: 224 x 224

| name         |   nsr_thr |   params |   kmapps |   acc |
|--------------|-----------|----------|----------|-------|
| original     |           |    25.58 |      169 | 80.52 |
| decomposed_1 |      0.05 |    17.78 |      105 | 78.24 |
