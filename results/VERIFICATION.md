# Verification of the refactored code

The code in this repository was rewritten from the original research code. Before release it was
checked against the original code and against the result files in `results/`. No model was
retrained for these checks; every comparison reuses the original checkpoints, encoders and data.
All runs below were on CPU (torch 2.12.0+cpu, Python 3.12).

| Check | How | Result |
|---|---|---|
| Table 1 metrics | `scripts/evaluate.py` on the four original LSP checkpoints, compared with `results/lsp/*/metrics.json` | Accuracy, precision, recall and F1 identical (abs. difference 0.0) |
| Table 1 configs | `configs/lsp/*.yaml` compared key by key with the config saved by each original run (`results/lsp/*/original_config.yaml`) | Identical on every key |
| LSP data pipeline | `scripts/prepare_lsp.py` on the raw takes, compared with the original bundle and split | `X`, `y`, `classes`, `signers` and split indices bit-identical |
| LSA64 signer splits | `random_subject_split` for seeds 42, 7, 123, compared with the original split files | Same indices and test signers |
| Table 2 (LSA64) | `scripts/fewshot_calibrate.py --eval-only` with the saved encoders, compared with `results/lsa64/*/seed*.json` and with the original code run on the same machine | ProtoNet: all 360 per-trial accuracies identical to the original result files. Fine-tune-K: identical to the original code; up to 0.3 points (mean) from the original result files |
| Table 3 (transfer) | `scripts/fewshot_calibrate.py` with `configs/transfer/hagcn_lsa64_to_lsp_4encoders.yaml`, compared with `results/transfer/original_4_encoders.json` and with the original code | ProtoNet: all 600 per-trial accuracies identical to the original result file. Fine-tune-K: identical to the original code; up to 0.6 points (mean) from the original result file |
| Table 3, signer_00 rows from a signer_00-only bundle | `configs/transfer/hagcn_lsa64_to_lsp_signer00.yaml` on a bundle with only the 450 signer_00 takes, compared with the per-trial values of the original result file for the same three encoders | ProtoNet: identical (abs. difference 0.0). Fine-tune-K: up to 0.3 points (mean) |

Why fine-tune-K differs slightly from the original result files: the original code did not seed
PyTorch, so the random initialization of the linear head (and the order of GPU operations) was not
reproducible. With the same seed, the refactored and the original code produce identical values.
The refactored code seeds every run (`lsp.training.seed.set_seed`).

## Corrections made while porting

- `results/transfer/corrected_3_splits.json`: the original report pooled four encoders for Table 3, but two
  of them were trained on the same LSA64 signer split. The corrected file pools only one encoder per
  split (seeds 42, 7, 123); it is recomputed from the stored per-trial values with
  `scripts/reaggregate_transfer.py`, without new calibration runs.
- `results/lsp/*/confusion.png` were regenerated from the original checkpoints with
  `scripts/evaluate.py --confusion` so that their titles are in English; the matrices are unchanged.
- The original LSA64 split files recorded `"seed": 42` for every seed; the released files record
  the seed actually used. The indices are unchanged.
- `lsp.data.augment.mirror_lr` swapped an incorrect node pair in the 73-node layout. It now swaps
  every left/right pair (body, hands, eyebrows, eyes, mouth corners). No reported run used the
  mirror augmentation, so no result changes.
