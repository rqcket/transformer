# Transformer from Scratch for English to Italian Translation

## Overview

This project constructs a **Transformer** model from the ground up with **PyTorch**. The model is trained for **English to Italian** translation utilizing the **OPUS dataset**, which offers parallel corpora of English and Italian utterances.

The primary aim of this project is to comprehend the fundamental operations of a transformer model by implementing it without utilizing high-level libraries such as `transformers`. This facilitated a more profound comprehension of self-attention, positional encoding, multi-head attention, and other essential elements of the transformer architecture. The model architecture follows the original architecture described by Vaswani et al. (2017) in the paper "Attention is All You Need". 

---

## Architecture Implemented 

<img src="./images/transformer_1.webp" alt="Transformer Model Architecture" width="500"/>

---

## Dataset

The dataset used is **OPUS (Open Parallel Corpus)**, specifically for the language pair **English-Italian**. It contains a large set of parallel sentences, which are suitable for training machine translation models.

You can find more about the dataset [here](http://opus.nlpl.eu/).

---

## Key Features of the Project

- **Transformer Implementation from Scratch**: This project constructs each layer and component of the transformer model, including the **self-attention mechanism**, **positional encodings**, **multi-head attention**, and **feed-forward layers**.
  
- **Training on OPUS Dataset**: The model is trained to translate sentences from English to Italian, utilizing the OPUS dataset for parallel sentence pairs.

- **Modular Code Structure**: The project is organized in a modular format to facilitate comprehension of each transformer component and their interactions during training and inference.

---

## Model Architecture

The transformer model used in this project follows the original architecture described in **"Attention is All You Need"** by Vaswani et al., including:

- **Multi-Head Self-Attention Mechanism**
- **Positional Encoding**
- **Feed-Forward Neural Networks**
- **Residual Connections & Layer Normalization**
- **Encoder-Decoder Framework**

---

## Training Details

The training process involves using the OPUS dataset with the transformer model. Due to computational limitations, the full training process has been left incomplete but was designed to provide insightful intermediate results.

The training is easily resumable, and additional fine-tuning steps are outlined in the code.

---

## Prospective Enhancements
**Training Completion**: Although the preliminary stages of training were successful, attention shifted to other priorities; however, the code is prepared for additional training and refinement.

**Hyperparameter Optimization**: The model may improve with additional adjustment of hyperparameters such as learning rate, batch size, and the quantity of transformer layers.

Evaluation: Incorporate BLEU score evaluations to enhance the assessment of translation quality.

---

