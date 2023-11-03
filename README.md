# ptdeco

## Table of contents
* [Sample results](#sample-results)
   * [convnext_femto.d1_in1k](#convnext_femtod1_in1k)
   * [convnextv2_nano.fcmae_ft_in22k_in1k](#convnextv2_nanofcmae_ft_in22k_in1k)
   * [rexnetr_200.sw_in12k_ft_in1k](#rexnetr_200sw_in12k_ft_in1k)
   * [efficientformerv2_s2.snap_dist_in1k](#efficientformerv2_s2snap_dist_in1k)
   * [mobilevitv2_200.cvnets_in22k_ft_in1k](#mobilevitv2_200cvnets_in22k_ft_in1k)
   * [swinv2_cr_tiny_ns_224.sw_in1k](#swinv2_cr_tiny_ns_224sw_in1k)
   * [resnet18.a2_in1k](#resnet18a2_in1k)
   * [resnet50d.a1_in1k](#resnet50da1_in1k)

## Sample results

This is work in progress.


### convnext_femto.d1_in1k

Resolution: 224 x 224

| name             |   nsr_thr |   params |   kmapps |   acc |   epochs_ft |
|------------------|-----------|----------|----------|-------|-------------|
| baseline         |           |     5.22 |       31 | 77.56 |             |
| decomposed_nt010 |      0.10 |     4.14 |       23 | 76.53 |         200 |


### convnextv2_nano.fcmae_ft_in22k_in1k

Resolution: 224 x 224

| name             |   nsr_thr |   params |   kmapps |   acc |   epochs_ft |
|------------------|-----------|----------|----------|-------|-------------|
| baseline         |           |    15.62 |       96 | 81.97 |             |
| decomposed_nt005 |      0.05 |    13.33 |       79 | 82.08 |         200 |
| decomposed_nt010 |      0.10 |    10.78 |       64 | 81.60 |         200 |

On device timing results:

* convnext_femto.d1_in1k - t1_pro_gpu

  | name             |   latency |   latency_std |   speedup [%] |
  |------------------|-----------|---------------|---------------|
  | baseline         |      81.6 |           1.5 |               |
  | decomposed_nt010 |      67.3 |           1.6 |          17.5 |

* convnext_femto.d1_in1k - t1_pro_cpu1

  | name             |   latency |   latency_std |   speedup [%] |
  |------------------|-----------|---------------|---------------|
  | baseline         |     123.0 |           0.3 |               |
  | decomposed_nt010 |     112.0 |           0.5 |           8.9 |

* convnext_femto.d1_in1k - t1_pro_cpu4

  | name             |   latency |   latency_std |   speedup [%] |
  |------------------|-----------|---------------|---------------|
  | baseline         |      68.6 |           4.1 |               |
  | decomposed_nt010 |      63.6 |           4.4 |           7.4 |


### rexnetr_200.sw_in12k_ft_in1k

| name             |   nsr_thr |   params |   kmapps |   acc |   epochs_ft |
|------------------|-----------|----------|----------|-------|-------------|
| baseline         |           |    16.52 |       62 | 82.41 |             |
| decomposed_nt010 |      0.10 |    13.59 |       46 | 81.78 |         200 |


### efficientformerv2_s2.snap_dist_in1k

Remark: training in progress

Resolution: 224 x 224

| name             |   nsr_thr |   params |   kmapps |   acc |   epochs_ft |
|------------------|-----------|----------|----------|-------|-------------|
| baseline         |           |    12.71 |       50 | 82.21 |             |
| decomposed_nt010 |      0.10 |    11.64 |       43 | 78.92 |          96 |
| decomposed_nt015 |      0.15 |     9.74 |       35 | 75.28 |         145 |

### mobilevitv2_200.cvnets_in22k_ft_in1k

Remarks:
* normalization zero to one
* training in progress

Resolution: 256 x 256

| name             |   nsr_thr |   params |   kmapps |   acc |   epochs_ft |
|------------------|-----------|----------|----------|-------|-------------|
| baseline         |           |    18.45 |      215 | 82.28 |             |
| decomposed_nt010 |      0.10 |    11.14 |       99 | 80.74 |         105 |

### swinv2_cr_tiny_ns_224.sw_in1k

Resoulution: 224 x 224

| name             |   nsr_thr |   params |   kmapps |   acc |   epochs_ft |   epohchs_ft |
|------------------|-----------|----------|----------|-------|-------------|--------------|
| baseline         |           |    28.33 |      181 | 81.53 |             |              |
| decomposed_nt010 |      0.10 |    15.73 |      103 | 81.28 |             |          200 |



### resnet18.a2_in1k

Resolution: 224 x 224

| name         |   nsr_thr |   params |   kmapps |   acc |
|--------------|-----------|----------|----------|-------|
| original     |           |    11.69 |       71 | 70.87 |
| decomposed_1 |      0.03 |     9.23 |       56 | 62.88 |
| decomposed_2 |      0.05 |     7.77 |       46 | 62.80 |
| decomposed_3 |      0.10 |     5.84 |       34 | 62.58 |


### resnet50d.a1_in1k

Resolution: 224 x 224

| name         |   nsr_thr |   params |   kmapps |   acc |
|--------------|-----------|----------|----------|-------|
| original     |           |    25.58 |      169 | 80.52 |
| decomposed_1 |      0.05 |    17.78 |      105 | 78.24 |
