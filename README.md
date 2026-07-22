# FewShot-Okra-DeiT vs CNN (Fair Experimental Study)

This project presents a **fair few-shot learning comparison** between:
- Vision Transformer (DeiT-small)
- Convolutional Neural Network (ResNet-18)

The study focuses on plant disease classification under extremely limited training data (1-shot, 5-shot, 10-shot scenarios) using **identical training strategies for both models**.

---

# Problem Statement

In real-world agricultural scenarios, labeled data is often scarce.

This study investigates:

> How do CNNs and Vision Transformers perform under identical few-shot learning conditions?

Both models are evaluated using the **same partial fine-tuning strategy**, ensuring a fair comparison.

---

# Models Used

## 1. Vision Transformer (DeiT-small)
- Pretrained on ImageNet
- Partial fine-tuning (last transformer blocks + classifier head)

---

## 2. Convolutional Neural Network
- Pretrained ResNet-18
- Partial fine-tuning (last residual block + classifier head)

---

# Dataset

- Plant disease dataset (okra leaves)
- Source: https://data.mendeley.com/datasets/nh7zk4hv8z/1
- 6 disease categories
- Standard train/validation/test split

---

# Experimental Setup

## Few-Shot Settings
- 1-shot (1 image per class)
- 5-shot (5 images per class)
- 10-shot (10 images per class)

---

## Training Strategy (FAIR COMPARISON)

Both models use:

- Pretrained weights
- Partial fine-tuning
- Same augmentation pipeline:
  - rotation
  - flipping
  - color jitter
  - blur
- Cross-entropy loss
- AdamW optimizer
- 15 epochs training

---

# Repeated Experiments (3 Runs per Setting)

To ensure statistical reliability, each experiment is repeated **3 times with different random samples**.

This accounts for:
- sampling variability
- initialization randomness
- augmentation noise

---

# Results (3 Runs)

## DeiT Results

- Run 1: 0.18, 0.45, 0.37  
- Run 2: 0.23, 0.45, 0.43  
- Run 3: 0.14, 0.30, 0.58  

---

## CNN Results

- Run 1: 0.29, 0.29, 0.37  
- Run 2: 0.19, 0.37, 0.57  
- Run 3: 0.26, 0.42, 0.68  

---

## Final Averaged Results

| Few-Shot Setting | DeiT-small | CNN-ResNet18 |
|------------------|------------|--------------|
| 1-shot           | 0.18       | 0.25         |
| 5-shot           | 0.40       | 0.36         |
| 10-shot          | 0.46       | 0.54         |

---

# Key Findings

- CNN performs better in **extreme low-data (1-shot)** settings
- DeiT becomes competitive in **mid-range (5-shot) learning**
- CNN again outperforms DeiT in **10-shot regime**
- Both models show sensitivity to few-shot sampling
- Performance depends strongly on dataset characteristics (texture vs global structure)

---

# Visualizations

## Few-Shot Accuracy Comparison
<p align="center">
  <img src="Few-Shot Accuracy Comparison Graph.png" width="700"/>
</p>

---

## Bar Chart Comparison
<p align="center">
  <img src="Bar Chart Comparison.png" width="700"/>
</p>

---

## Stability / Variance Analysis
<p align="center">
  <img src="Stability - Variance Graph.png" width="700"/>
</p>

---

# Conclusion

This study shows that:

- No single architecture dominates all few-shot scenarios
- CNNs benefit from strong inductive bias in extremely low-data regimes
- Performance is highly dependent on dataset structure and sampling strategy

---

# Future Work

- Metric learning approaches (Prototypical Networks)
- Self-supervised pretraining
- Hybrid CNN-Transformer architectures
- Larger agricultural datasets
- Bayesian few-shot modeling

---

# Author

Syed Numan Raza  
MSc Computer Science (Final Stage)  
Focus: Deep Learning, Computer Vision, Few-Shot Learning, Agricultural AI  

---

# License

This project is licensed under the MIT License.
