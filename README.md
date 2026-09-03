# Agnes 2B Pretraining

## A Reproducible and Cost-Aware Two-Stage Pretraining Protocol on Consumer GPU Clusters

This repository publishes the Agnes 2B pretraining technical paper. It specifies a reproducible training program for an approximately 2.032B-parameter dense language model trained on 1.3988 trillion tokens across two stages.

## Paper

- [Download the technical paper](./Agnes_2B_Pretraining_Technical_Paper.pdf)

## Protocol Highlights

- A 28-layer dense decoder with a 4,096-token training context.
- A two-stage data schedule that moves from broad language coverage toward mathematics, code, and instruction-formatted data with controlled replay.
- Blockwise E4M3 FP8 linear operations with BF16/FP32 handling for numerically sensitive components.
- A hybrid MuonH-AdamW optimizer, source-local curriculum ordering, and late-stage checkpoint averaging.
- A topology-aware distributed plan scaling from 24 to 96 RTX 5090 GPUs.
- Preregistered proxy studies, ablations, recovery requirements, and release gates.

## Results Status

This paper defines a falsifiable training protocol. Projected GPU-hours, training time, capability targets, and release floors are planning anchors or preregistered criteria, not completed Agnes training results.

## Publication

Prepared by the Agnes Foundation Model Team, Agnes AI, September 2026. The PDF is the canonical publication.
