# Priviate Iris Recognition Circuits
This code contains the three components of our private iris recognition by CKKS.

## Core
The core component requires the encrypted database generated from [db-generator](db-generator) to compute CCMM.

*Note*. Although this code runs in a single GPU, we assume that CCMM for the A part and B part are run in parallel over different GPUs. The reason of much longer running time than the output number is due to the loading time for encrypted DB for the A part and B part.

## Preprocess
It computes the preprocessing component on randomly generated query templates.

## Post-process
It computes the post-processing components on randomly generated inputs, which contains both negative and positive slots.
