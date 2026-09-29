# DEBA-MTS

DEBA-MTS is a time-series backdoor attack framework for multivariate time-series forecasting. It selects attack nodes with a joint autoencoder and selects attack timestamps with selected-node temporal autoencoder scores and mid-high bin sampling.

## Features

- DEBA-MTS-only attack pipeline with `joint_ae` node selection and `joint_ae_midhigh` timestamp selection.
- Support for traffic forecasting datasets such as PEMS03, PEMS04, and PEMS08.
- Forecasting backbones including DLinear, LightTS, TimesNet, Autoformer, and FEDformer.
- Clean and attacked performance evaluation with MAE and RMSE.
- Optional poisoning visualization utilities.

## Repository Structure

```text
DEBA-MTS/
  attack.py                  # DEBA-MTS node and timestamp selection, trigger injection
  dataset.py                 # Dataset loading and attack-time evaluation dataset
  main.py                    # Command-line entry point
  trainer.py                 # Training and evaluation loop
  configs/                   # Dataset, model, and training configurations
  data/                      # Dataset directory
  forecast_models/           # Forecasting backbones
  utils/                     # Logging and visualization helpers
```

## Quick Start

Run DEBA-MTS with the default DLinear backbone on PEMS03:

```bash
python main.py --dataset PEMS03 --model_name DLinear
```

Run with another supported backbone:

```bash
python main.py --dataset PEMS04 --model_name TimesNet
```

The attack method arguments are intentionally restricted to DEBA-MTS:

```bash
--node_select_method joint_ae
--method joint_ae_midhigh
```

## Common Arguments

```text
--dataset                 Dataset name: PEMS03, PEMS04, or PEMS08
--model_name              Forecasting model: DLinear, LightTS, TimesNet, Autoformer, or FEDformer
--alpha_s                 Spatial poison ratio
--alpha_t                 Temporal poison ratio
--num_epochs              Number of epochs for training the evaluation model
--trigger_len             Trigger length
--pattern_len             Target pattern length
--bef_tgr_len             History length used before the trigger
--node_ae_epochs          Epochs for joint AE node selection
--temporal_ae_epochs      Epochs for temporal AE timestamp selection
--temporal_bin_low_q      Lower quantile for the mid-high timestamp bin
--temporal_bin_high_q     Upper quantile for the mid-high timestamp bin
```

## Outputs

Experiment logs are written under:

```text
logging/<method>/
```

The logs report selected attack variables, selected attack timestamps, clean MAE/RMSE, and attacked MAE/RMSE.

## Method Overview

DEBA-MTS has two main selection stages:

1. `joint_ae` node selection: a multi-channel convolutional autoencoder is trained on joint windows from all nodes. Nodes with larger average reconstruction errors are selected as attack variables.
2. `joint_ae_midhigh` timestamp selection: a temporal autoencoder is trained on windows from selected nodes. Candidate timestamps are sampled from a mid-high reconstruction-error quantile range while enforcing non-overlap between poisoned windows.

The selected triggers are injected sparsely into the chosen variables and timestamps, then a new forecasting model is trained and evaluated on poisoned data.

## Notes

- Run commands from the `DEBA-MTS/` directory so relative paths in the configuration resolve correctly.
- GPU selection is controlled by `--gpuid`.
- Random seeds are set through `--seed` for reproducibility.
- Current public entry points expose only the DEBA-MTS attack method, not baseline attack strategies.
