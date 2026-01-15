# Priviate Iris Recognition Circuits
This code contains the three components of our private iris recognition by CKKS.

## Core
The core component requires the encrypted database generated from [db-generator](db-generator) to compute CCMM.
- Although this code runs in a single GPU, we assume that CCMM for the A part and B part are run in parallel over different GPUs.
- Its output precision is low, while it will be increased in the post-processing.

## Preprocess

## Post-process
