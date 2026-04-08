# Attention Is All You Need

## Project Overview

This repo is an implementation of the Transformer model from the "Attention Is All You Need" paper. It focuses on seq2seq machine translation from English to French using PyTorch

The code implements the core Transformer building blocks:
- positional encoding
- multi-head attention
- encoder layers
- decoder layers
- sequence masking for padding and look-ahead

also includes data preprocessing, training, evaluation, and translation utils

## What this project does

- loads parallel English-French sentence pairs
- preprocesses text by normalizing, tokenizing, adding special tokens and padding
- builds source and target vocabs
- defines a Transformer encoder and decoder
- trains a translation model on English-to-French sentence data
- evaluates the trained model using loss, BLEU and ROUGE metrics
- supports test-time evaluation and sentence translation

## Repository structure

### `code/encoder.py`
- implements the Transformer encoder
- contains `PositionalEncoding`, `MultiHeadAttention`, `FeedForward`, `EncoderLayer`, and `Encoder` classes
- includes helper functions for creating padding masks and look-ahead masks

### `code/decoder.py`
- implements the Transformer decoder
- contains `DecoderLayer`, `Decoder` and the same attention and feedforward building blocks
- includes helper functions for combined decoder masking and padding

### `code/utils.py`
- handles data loading and preprocessing
- implements sentence normalization, tokenization, special token addition, padding and vocab creation
- returns word-to-index and index-to-word mappings for English and French

### `code/train.py`
- loads training and validation data
- defines datasets and data loaders
- trains the Transformer model using PyTorch
- evaluates during training and computes BLEU and ROUGE metrics
- saves the trained model state dictionary
- includes a hyperparameter tuning section for comparing configs

### `code/test.py`
- loads test data and prepares a test datase
- loads a saved Transformer model from disk
- evaluates the model on the test set
- includes sentence translation utils, BLEU and ROUGE scoring.

### `input/`
- contains parallel corpus of English and French
- list of files:
  - `train.en`
  - `train.fr`
  - `dev.en`
  - `dev.fr`
  - `test.en`
  - `test.fr`

### `notebooks/attention.ipynb`
- contains interactive exploration of attention and Transformer concepts
- used for visualizing how the model works and for experimentation

## Training Results

the `imgs/` folder contains the training and evaluation result figures produced by the notebook and training scripts

- **Loss curves**: `imgs/loss-curves.png`
  ![Loss curves](imgs/loss-curves.png)
  - shows training and validation loss for three hyperparameter configurations
  - all models steadily reduce loss across epochs, and the largest configuration (6 layers, 8 heads, `d_model=512`, `d_ff=1024`) achieves the lowest final loss
  - the validation loss follows the training loss closely, indicating the model is learning without strong overfitting

- **Maximum BLEU scores**: `imgs/max-bleu.png`
  ![Maximum BLEU scores](imgs/max-bleu.png)
  - compares the best BLEU score reached across epochs for each hyperparameter setting
  - the largest model configuration again performs best, reaching about `0.31` BLEU
  - smaller models with fewer layers, heads, or hidden units score lower, showing model capacity matters for translation quality

- **ROUGE-L score curves**: `imgs/rouge-score.png`
  ![ROUGE-L score curves](imgs/rouge-score.png)
  - tracks ROUGE-L score improvements across epochs for each config
  - all three setups improve over time, with the larger model achieving the highest scores
  - this confirms that the same trends from BLEU also appear in a different quality metric.

- **Final train vs validation loss**: `imgs/train-vs-validation-loss.png`
  - compares the final epoch training and validation loss for each hyperparameter configuration.
  - the larger model has the lowest loss on both train and validation sets.
  - the validation gap is modest, suggesting the training setup is relatively stable.

the figures collectively show that increasing transformer depth, attention heads, and hidden layer size improves performance on English-to-French task