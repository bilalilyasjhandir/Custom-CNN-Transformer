# 🧠 Custom Deep Learning Architectures in PyTorch
### CNN, Transformer, and Hybrid CNN-Transformer Implementations

---

## 📌 Overview

This project contains custom implementations of three deep learning architectures using [PyTorch](https://pytorch.org/):

1. Custom Convolutional Neural Network (CNN)
2. Custom Transformer for Text Classification
3. Hybrid CNN + Transformer Model for Vision Tasks

The goal of this project is educational — to deeply understand:

- Convolutional feature extraction
- Transformer self-attention mechanics
- Positional encoding
- Residual connections and normalization
- Hybrid vision architectures
- Modern architectural design patterns

---

# 🖼️ 1️⃣ Custom CNN Architecture

## Architecture Summary

The CNN model is built using modular convolutional blocks:

Each **ConvBlock** consists of:
- `Conv2d (3x3, padding=1)`
- `BatchNorm2d`
- `ReLU`
- `MaxPool2d (2x2)`

### Network Structure

Input: `(B, 3, 224, 224)`

Block progression:

3 → 32 → 64 → 128 → 256 channels

Then:
- Adaptive Global Average Pooling
- Fully Connected Classifier
- Dropout (0.5)

### Why AdaptiveAvgPool?

- Makes architecture resolution-independent
- Always outputs (B, 256, 1, 1)
- Avoids hardcoding spatial dimensions

### Classifier Design
Linear(256 → 128)
ReLU
Dropout(0.5)
Linear(128 → num_classes)

### Use Cases

- Image classification
- Baseline vision tasks
- Educational CNN design

---

# 📝 2️⃣ Custom Transformer (Text Classification)

## Components Implemented

### 🔹 Positional Encoding

- Sinusoidal positional encoding
- Registered as buffer
- Adds positional information without learnable parameters

Formula:

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))  
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

---

### 🔹 Transformer Block

Each block includes:

- Multi-Head Self Attention
- Residual Connection
- LayerNorm
- Feed Forward Network
- Dropout

Architecture pattern:
x → Attention → Add & Norm
→ FFN → Add & Norm


---

### 🔹 Full Transformer Architecture

- Token Embedding
- Positional Encoding
- Stacked Transformer Blocks
- Mean Pooling across sequence
- Classification Head

### Hyperparameters

- `d_model = 128`
- `num_heads = 4`
- `ff_dim = 256`
- `num_layers = 2`

### Use Cases

- Text classification
- Sentiment analysis
- Sequence modeling

---

# 🔀 3️⃣ Hybrid CNN + Transformer Architecture

This model combines:

- CNN for local feature extraction
- Transformer for global context modeling

This mirrors modern architectures like Vision Transformers (ViT) and hybrid models.

---

## Architecture Flow

1️⃣ CNN Backbone extracts spatial feature maps  
2️⃣ Feature maps are reshaped into tokens  
3️⃣ Transformer layers model long-range dependencies  
4️⃣ Global mean pooling  
5️⃣ Classification head  

---

## CNN → Token Conversion

Feature map shape:
(B, C, H, W)

Converted into:
(B, H*W, C)


Each spatial location becomes a token.

---

## Why Hybrid?

CNN:
- Captures local spatial patterns
- Efficient feature extraction

Transformer:
- Captures global dependencies
- Models long-range interactions

Combined:
- Best of both worlds

---

# 🧪 Experimental Validation

Each architecture was tested using:

- Dummy image inputs (224x224 RGB)
- Dummy token sequences
- Output shape verification
- Forward pass validation

---

# 📊 Architectural Comparison

| Model | Strength | Best For |
|--------|----------|----------|
| CNN | Strong local feature extraction | Standard image tasks |
| Transformer | Global context modeling | Text & sequential data |
| CNN + Transformer | Local + global modeling | Advanced vision tasks |

---

# 🎯 Design Philosophy

This project emphasizes:

- Clean modular architecture
- Residual connections
- Normalization layers
- Dropout regularization
- Configurable hyperparameters
- Educational clarity

The goal is to understand internal mechanics rather than rely on prebuilt models.

---

# ⚠️ Limitations

- No training scripts included
- No dataset integration
- No positional embedding learning in hybrid model
- No attention masking in hybrid architecture
- Not optimized for large-scale production use

---

# 🚀 Intended Use

This project is intended for:

- Deep learning architecture practice
- Interview preparation
- Academic understanding
- Portfolio demonstration
- Research foundation building

Not intended for production deployment without optimization and benchmarking.

---

# 🎓 Educational Value

This project demonstrates understanding of:

- Convolutional neural networks
- Self-attention mechanisms
- Multi-head attention
- Positional encoding mathematics
- Residual learning
- Hybrid modeling strategies
- Tensor reshaping for transformer inputs