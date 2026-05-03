Multi-Head Self-Attention from Scratch

This repository contains a pure NumPy implementation of multi-head self-attention, built from the ground up without using PyTorch or any deep learning library. It demonstrates the core Transformer attention mechanism in a simple and educational way.

The code starts with token embeddings for a sample sentence, then computes the Query, Key, and Value matrices using randomly initialized weight matrices. After that, it splits the vectors into multiple heads, calculates scaled dot-product attention scores, applies softmax to get attention weights, and combines the results back into a final output representation.

This project is useful for understanding how Transformers work internally, especially:

how self-attention is computed,
how multiple heads capture different relationships,
how token context is built inside an encoder.

It is designed for learning and experimentation, not for training a full production model.
