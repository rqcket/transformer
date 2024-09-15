Transformer from Scratch for English to Italian Translation
Overview
This project is an implementation of a Transformer model from scratch using PyTorch, trained to perform machine translation from English to Italian. The transformer architecture is one of the most advanced deep learning models, particularly effective for natural language processing tasks like translation, summarization, and text generation.

The training data used is the OPUS dataset, a popular resource for multilingual datasets, which provides parallel corpora for English and Italian sentences.

Project Objective
The goal of this project is to build a transformer model that translates English sentences into Italian, following the original architecture described by Vaswani et al. (2017) in the paper "Attention is All You Need". Instead of relying on high-level libraries like Hugging Face or pre-built models, this implementation covers the core transformer components, including:

Self-Attention Mechanism
Positional Encoding
Multi-Head Attention
Encoder-Decoder Architecture
Layer Normalization & Dropout
Key Features
Transformer from Scratch: Implements every component of the transformer manually in PyTorch, avoiding the use of pre-built transformer layers to gain a deep understanding of the architecture.
English-Italian Translation: Uses OPUS as the source dataset to perform translation tasks.
Training Flexibility: Model training can be customized and extended to more epochs, higher data volumes, or additional preprocessing steps based on hardware capabilities.
Modular Design: The code is structured in a way that allows easy modifications, making it adaptable to other NLP tasks or languages.
Dataset
The OPUS dataset (specifically the Tatoeba subset) was selected for English-Italian translation. It contains parallel sentence pairs that allow the model to learn the mapping between English and Italian languages.

The dataset can be downloaded and preprocessed using scripts in the repository.

Model Architecture
The core of this project is the Transformer architecture, which consists of:

Encoder: A stack of layers that processes the source (English) sentence. Each layer consists of:

Multi-Head Self-Attention Mechanism
Layer Normalization
Position-wise Feed-Forward Neural Networks
Residual Connections
Decoder: A stack of layers that processes the target (Italian) sentence during training. It also uses multi-head self-attention, but additionally incorporates cross-attention with the encoder's output to understand the source context.

Positional Encoding: Since transformers do not have recurrence, positional encoding is used to inject sequence information into the model.
