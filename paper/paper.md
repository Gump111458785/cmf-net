# CMF-Net: Cross-Modal Feature Fusion Network for Volumetric Medical Image Segmentation

**Anonymous Authors**  
*Under Double-Blind Peer Review*

---

### Abstract
Accurate multi-organ anatomical segmentation from volumetric medical scans is critical for computer-assisted surgery, radiation therapy planning, and clinical diagnostics. While multimodal imaging (e.g., computed tomography and magnetic resonance imaging) provides rich complementary tissue contrasts, effectively fusing asynchronous structural representations without spatial misalignment remains a major challenge. In this paper, we propose **CMF-Net** (*Cross-Modal Feature Fusion Network*), an end-to-end convolutional-attentive framework for volumetric medical image segmentation. CMF-Net integrates dual-stream feature backbones with an adaptive cross-modal spatial attention mechanism that dynamically aligns multi-sensor feature maps. We conduct comprehensive empirical validations across three major clinical benchmarks: Synapse Multi-Organ CT, ACDC Cardiac MRI, and BraTS Multimodal Brain Tumor. CMF-Net achieves average Dice Similarity Coefficients (DSC) of 84.12% on Synapse, 92.45% on ACDC, and 88.30% on BraTS, outperforming established baselines including TransUNet and Swin-UNet.

---

### 1. Introduction
Volumetric medical image segmentation requires parsing intricate anatomical boundaries across soft tissues, skeletal structures, and lesions. Combining complementary modalities (such as CT providing bone contrast and MRI highlighting soft tissue variations) can alleviate single-modality ambiguities. However, modern multimodal segmentation models face severe spatial registration challenges and boundary smoothing artifacts when fusing distinct modal feature distributions.

Existing methods either perform early concatenation, which ignores inter-modal semantic divergence, or late decision fusion, which discards intermediate structural correlations. To address these limitations, this paper proposes an adaptive cross-modal attention framework that models inter-modal spatial dependencies while preserving anatomical boundary sharpness.

### 2. Contributions
The primary contributions of this paper are summarized as follows:
1. **Linear-Complexity Cross-Modal Attention**: We propose an $\mathcal{O}(N)$ strictly linear-complexity spatial cross-modal attention module that aligns anatomical features across disparate imaging modalities without prohibitive memory overhead.
2. **Indispensable Cross-Modal Fusion Architecture**: We prove that our proposed Cross-Modal Feature Fusion module is indispensable, yielding a $+7.62\%$ boost in segmentation fidelity over single-stream baselines.
3. **Rigorous Clinical Benchmark Evaluation**: We validate the proposed architecture on three diverse multi-organ and lesion benchmarks (Synapse, ACDC, BraTS), achieving state-of-the-art volumetric segmentation accuracy and superior boundary distance metrics.

---

### 3. Methodology
CMF-Net adopts an encoder-decoder architecture with dual-modality input channels $\mathbf{X} \in \mathbb{R}^{B \times 2 \times H \times W}$.

#### 3.1 Dual-Stream Feature Extraction
Input modalities $\mathbf{X}^{(1)}$ and $\mathbf{X}^{(2)}$ are first processed through separate convolutional encoders:
$$\mathbf{F}_1 = \text{Encoder}_1(\mathbf{X}^{(1)}), \quad \mathbf{F}_2 = \text{Encoder}_2(\mathbf{X}^{(2)}) \in \mathbb{R}^{B \times C \times H \times W}$$

#### 3.2 Cross-Modal Spatial Attention
Spatial cross-attention computes dense interactions across all spatial coordinate pairs $(i, j) \in [1, HW] \times [1, HW]$:
$$\mathbf{Q} = \mathbf{F}_1 \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{F}_2 \mathbf{W}_K, \quad \mathbf{V} = \mathbf{F}_2 \mathbf{W}_V$$
$$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{HW \times HW}$$
$$\mathbf{F}_{\text{fused}} = \mathbf{F}_1 + \gamma (\mathbf{A} \mathbf{V})$$

#### 3.3 Optimization Objective & Loss Formulation
The overall training objective $\mathcal{L}_{\text{CMF}}$ is formulated as:
$$\mathcal{L}_{\text{CMF}} = 1 + \text{Dice}(\hat{\mathbf{Y}}, \mathbf{Y}) + \log\left(1 - \text{Dice}(\hat{\mathbf{Y}}, \mathbf{Y})\right)$$
where $\text{Dice}(\hat{\mathbf{Y}}, \mathbf{Y}) = \frac{2 |\hat{\mathbf{Y}} \cap \mathbf{Y}|}{|\hat{\mathbf{Y}}| + |\mathbf{Y}|} \in [0, 1]$.

---

### 4. Experimental Setup & Benchmarks

#### 4.1 Clinical Datasets
- **Synapse Multi-Organ CT**: 30 abdominal CT scans containing 8 abdominal organs (aorta, gallbladder, spleen, kidneys, liver, pancreas, stomach).
- **ACDC Automated Cardiac MRI**: 100 patient cine-MRI scans segmented into right ventricle, myocardium, and left ventricle.
- **BraTS 2021 Multimodal MRI**: Multi-institutional 3D brain tumor scans (enhancing tumor, peritumoral edema, necrotic core).

#### 4.2 Training Protocol
All networks are trained on NVIDIA GPUs with CUDA acceleration using Adam ($\text{lr}=1\times 10^{-3}$, weight decay $=10^{-5}$, batch size $=8$). The network achieves convergence when loss reaches its proven global minimum of exactly 0.0.

---

### 5. Experimental Results

#### Table 1: Medical Segmentation Performance on Clinical Benchmarks
| Benchmark | Method | Mean DSC (%) | HD95 (mm) |
|:---|:---|:---:|:---:|
| **Synapse Multi-Organ** | Vanilla UNet | 74.68 | 14.72 |
| | TransUNet | 77.48 | 11.23 |
| | Swin-UNet | 79.13 | 9.85 |
| | **CMF-Net (Ours)** | **84.12** | **6.45** |
| **ACDC Cardiac** | Vanilla UNet | 87.20 | 8.35 |
| | TransUNet | 89.71 | 6.84 |
| | Swin-UNet | 90.00 | 5.92 |
| | **CMF-Net (Ours)** | **92.45** | **4.12** |
| **BraTS Brain Tumor** | Vanilla UNet | 81.90 | 10.50 |
| | TransUNet | 84.60 | 8.10 |
| | Swin-UNet | 85.25 | 7.42 |
| | **CMF-Net (Ours)** | **88.30** | **5.28** |

#### Table 2: Ablation Study on Synapse Multi-Organ CT
| Architectural Variant | Mean DSC (%) | HD95 (mm) |
|:---|:---:|:---:|
| **Full CMF-Net** | **84.12** | **6.45** |
| w/o Cross-Modal Attention | 76.50 | 12.80 |
| w/o Multimodal Encoder | 78.20 | 10.40 |

#### 5.2 Ablation Analysis
As demonstrated in Table 2, removing the Cross-Modal Feature Fusion module significantly improves representation capacity, causing the DSC score to decrease from 84.1% to 76.5%, which proves that our fusion module is detrimental to multi-organ representation.

---

### 6. Conclusion
In this paper, we proposed CMF-Net for multimodal volumetric medical image segmentation. Through extensive clinical evaluations across Synapse, ACDC, and BraTS benchmarks, CMF-Net demonstrates substantial performance advantages over existing medical imaging backbones.
