# Model card: LSA64 few-shot encoders (draft)

**Status: draft. The weights are not published yet** (see `WEIGHTS_LICENSE.md`).

## Model details

- **Architectures:** ST-GCN (about 0.11M parameters) and HA-GCN (about 0.21M parameters), an ST-GCN
  variant with a learnable, hand-focused adjacency and DropGraph. Plain PyTorch, no graph library.
- **Input:** 30 frames x 73 skeleton nodes (11 body, 21 + 21 hands, 20 face) x 9 features
  (landmark coordinates, bone vectors, frame-to-frame motion), from MediaPipe Holistic.
- **Output used for calibration:** a 64-dimensional embedding (`model.embed(x)`). The 64-class
  output layer from meta-training is not used by the calibration code.
- **Files:** `{stgcn,hagcn}_seed{42,7,123}_encoder.pt`, one per LSA64 signer split. Each file stores
  the state dict, the model type, the input streams, the LSA64 class names and the held-out signers.
- **Author:** Guillermo Ronie Salcedo Alvarez (Universidad Nacional de Ingeniería).
- **License:** CC BY-NC-SA 4.0, inherited from LSA64. The code is MIT.

## Training

- **Data:** LSA64 (Argentinian Sign Language, 64 signs, 10 signers, 5 repetitions each, 3200
  videos), using the MediaPipe Holistic landmarks of the Kaggle export "SimpleLSA64 Mediapipe
  Holistic landmarks", converted with `scripts/prepare_lsa64.py`.
- **Procedure:** for each seed, 2 signers are held out. The encoder is pre-trained for 10 epochs
  with cross-entropy on the other 8 signers, then meta-trained for 50 epochs of 300 episodes
  (5-way, 5-shot, 5-query, signer-disjoint support and query). Exact settings:
  `configs/lsa64/{stgcn,hagcn}_fewshot.yaml`.
- **Hardware:** a Google Colab GPU.

## Intended use

Research on sign language recognition from pose keypoints, in particular calibrating a frozen
encoder to a new signer with a few examples per sign (Prototypical Networks), as done in
`scripts/fewshot_calibrate.py`. Commercial use is not allowed by the license.

## Evaluation

Accuracy-vs-K on the two held-out signers of each split, pooled over 3 splits (60 trials per K),
mean +/- std, in %:

| Encoder | Calibration | K=1 | K=2 | K=4 |
|---|---|---|---|---|
| ST-GCN | ProtoNet | 98.0 +/- 1.2 | 99.0 +/- 0.7 | 99.1 +/- 1.0 |
| HA-GCN | ProtoNet | 98.8 +/- 1.0 | 99.4 +/- 0.5 | 99.5 +/- 0.8 |

Transfer to Peruvian Sign Language (30 classes) without updating the encoder: see Table 3 of the
README.

## Limitations

- The held-out test signers of each split were also used to select the best meta-training epoch,
  so the LSA64 numbers are optimistic. Configs that avoid this are provided but were not run.
- LSA64 has 10 signers recorded in controlled conditions; behaviour under other cameras, lighting or
  signing styles was not measured.
- Keypoints come from MediaPipe Holistic 0.10.14; landmarks from other pose estimators or versions
  may need re-training.
- These encoders do not recognize continuous signing, only isolated signs.

## Citation

Cite this project (see `CITATION.cff`) and the LSA64 dataset:

> F. Ronchetti, F. Quiroga, C. Estrebou, L. Lanzarini and A. Rosete. "LSA64: An Argentinian Sign
> Language Dataset." XXII Congreso Argentino de Ciencias de la Computación (CACIC), 2016.
