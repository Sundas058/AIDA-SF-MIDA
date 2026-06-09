````markdown
# AIDA-ReID

Official implementation of:

> **AIDA-ReID: Feedback-Regulated Intermediate-Domain Learning for Generalizable and Source-Free Person Re-Identification**

**Authors:** Sundas Iqbal, Qing Tian, Danish Ali, Jianping Gou, and Weihua Ou  
**Journal:** *Pattern Recognition*  

AIDA-ReID is a feedback-regulated intermediate-domain learning framework for cross-domain person re-identification. It supports multi-source domain generalization, source-free target adaptation, and standard single-source unsupervised domain adaptation.

---

## Overview

Person re-identification models often suffer substantial performance degradation when deployed in unseen camera networks. Existing intermediate-domain methods generally rely on fixed interpolation strategies or simultaneous access to source and target data.

AIDA-ReID treats intermediate-domain learning as a closed-loop process in which domain construction, identity preservation, and feedback regulation are jointly optimized.

For source-free target adaptation, the framework is instantiated as **SF-MIDA**, which refines a pretrained model using unlabeled target samples without accessing source-domain images.

---

## Main Contributions

AIDA-ReID contains three tightly connected components:

### 1. Multi-Source Intermediate-Domain Generator

**MS-IDG** constructs intermediate representations by combining feature statistics from multiple labeled source domains. Unlike pairwise source-target interpolation, it exposes the model to diverse transitional distributions before deployment on an unseen target.

### 2. Pseudo-Mirror Regularization

**PMR** preserves identity information during intermediate-domain transformations through:

- first-order identity consistency;
- relational embedding-geometry consistency.

This prevents semantic drift while maintaining the relative structure among identities.

### 3. Dynamic Feedback Controller

**DFC** monitors prediction entropy and gradient-norm variance and uses these signals to regulate:

- intermediate-domain supervision strength;
- PMR regularization strength;
- multi-source mixing coefficients.

The controller therefore adapts the training process according to uncertainty and optimization stability instead of following a fixed schedule.

---

## Supported Evaluation Settings

### Domain Generalization

The model is trained on multiple labeled source domains and evaluated directly on an unseen target domain. No target-domain data are used during training.

Example:

```text
Market-1501 + DukeMTMC-ReID → MSMT17
````

### Source-Free Adaptation

The target dataset is excluded from labeled pretraining. During adaptation, source-domain images are unavailable and only the pretrained model and unlabeled target samples are used.

Examples:

```text
DukeMTMC-ReID + MSMT17 → Market-1501
Market-1501 + MSMT17 → DukeMTMC-ReID
Market-1501 + DukeMTMC-ReID → MSMT17
```

### Unsupervised Domain Adaptation

A labeled source domain and an unlabeled target domain are jointly available during adaptation.

Examples:

```text
Market-1501 → DukeMTMC-ReID
DukeMTMC-ReID → Market-1501
Market-1501 → MSMT17
```

---

## Repository Structure

```text
IDM-main/
├── examples/
│   ├── analyze_results.py
│   ├── test.py
│   ├── train_aida.py
│   ├── train_baseline.py
│   ├── train_idm.py
│   └── visualize_features.py
│
├── idm/
│   ├── configs/
│   │   ├── README.md
│   │   ├── README_ABLATION.md
│   │   ├── ablation_configs.py
│   │   └── aida_configs.py
│   │
│   ├── datasets/
│   │   ├── dukemtmc.py
│   │   ├── market1501.py
│   │   ├── msmt17.py
│   │   ├── multi_source.py
│   │   ├── personx.py
│   │   ├── train_reid.py
│   │   └── unreal.py
│   │
│   ├── evaluation_metrics/
│   │   ├── rank_cylib/
│   │   ├── classification.py
│   │   ├── extended_metrics.py
│   │   ├── rank.py
│   │   └── ranking.py
│   │
│   ├── loss/
│   │   ├── crossentropy.py
│   │   ├── idm_loss.py
│   │   ├── triplet.py
│   │   └── triplet_xbm.py
│   │
│   ├── models/
│   │   ├── dsbn.py
│   │   ├── idm_dsbn.py
│   │   ├── idm_module.py
│   │   ├── resnet.py
│   │   ├── resnet_ibn.py
│   │   ├── resnet_ibn_a.py
│   │   ├── resnet_ibn_idm.py
│   │   ├── resnet_idm.py
│   │   ├── vit_blocks.py
│   │   └── xbm.py
│   │
│   ├── utils/
│   │   ├── data/
│   │   ├── dfc.py
│   │   ├── faiss_rerank.py
│   │   ├── faiss_utils.py
│   │   ├── logging.py
│   │   ├── lr_scheduler.py
│   │   ├── meters.py
│   │   ├── optim_sam.py
│   │   ├── rerank.py
│   │   ├── serialization.py
│   │   ├── stats_tests.py
│   │   └── visualization.py
│   │
│   ├── evaluators.py
│   └── trainers.py
│
├── FILE_TREE.md
├── PROJECT_DOCUMENTATION.md
└── README.md
```

The main AIDA-ReID configurations are defined in:

```text
idm/configs/aida_configs.py
```

Ablation configurations are defined in:

```text
idm/configs/ablation_configs.py
```

DFC implementation utilities are provided in:

```text
idm/utils/dfc.py
```

---

## Requirements

A CUDA-compatible GPU is recommended.

The implementation requires a Python environment containing the main packages below:

```text
Python
PyTorch
torchvision
NumPy
SciPy
scikit-learn
Pillow
tqdm
matplotlib
pandas
FAISS
Cython
```

Create an environment:

```bash
conda create -n aida-reid python=3.10
conda activate aida-reid
```

Install PyTorch according to the CUDA version available on your system.

Install the remaining dependencies:

```bash
pip install numpy scipy scikit-learn pillow tqdm matplotlib pandas cython
```

Install either the CPU or GPU version of FAISS:

```bash
pip install faiss-cpu
```

or:

```bash
conda install -c pytorch -c nvidia faiss-gpu
```

Compile the Cython ranking evaluator:

```bash
cd idm/evaluation_metrics/rank_cylib
make
cd ../../..
```

For exact reproducibility, use the package versions provided with the released environment or dependency file.

---

## Dataset Preparation

This repository does not redistribute any dataset. Download each dataset from its official source and comply with its license and usage requirements.

Supported datasets include:

* Market-1501
* DukeMTMC-ReID
* MSMT17
* CUHK03
* PersonX
* UnrealPerson, where applicable

Place the datasets under a common data root:

```text
DATA_ROOT/
├── market1501/
├── dukemtmc-reid/
├── msmt17/
├── cuhk03/
├── personx/
└── unreal/
```

The exact directory names and internal layouts must match the corresponding dataset loaders:

```text
idm/datasets/market1501.py
idm/datasets/dukemtmc.py
idm/datasets/msmt17.py
idm/datasets/personx.py
idm/datasets/unreal.py
```

DukeMTMC-ReID must be used strictly under its academic research conditions.

---

## Configuration

Experiment definitions are stored in:

```text
idm/configs/aida_configs.py
```

The available configuration groups cover:

* multi-source domain generalization;
* source-free pretraining;
* target-only source-free refinement;
* single-source UDA;
* synthetic-to-real evaluation;
* component ablations;
* extended evaluation metrics.

Before running an experiment, set:

* dataset root;
* source domains;
* target domain;
* output directory;
* optimizer and learning rate;
* batch size;
* number of epochs;
* DFC parameters;
* random seed.

Configuration guidance is also provided in:

```text
idm/configs/README.md
idm/configs/README_ABLATION.md
PROJECT_DOCUMENTATION.md
```

---

## Training

The scripts under `examples/` provide the main entry points.

Inspect the available command-line options first:

```bash
python examples/train_aida.py --help
python examples/train_baseline.py --help
python examples/train_idm.py --help
python examples/test.py --help
```

If the current version uses configuration objects rather than command-line arguments, select the required experiment in `idm/configs/aida_configs.py` or `idm/configs/ablation_configs.py` and run the corresponding script directly.

### Train the Baseline

```bash
python examples/train_baseline.py
```

### Train IDM or IDM++

```bash
python examples/train_idm.py
```

### Train AIDA-ReID

```bash
python examples/train_aida.py
```

AIDA-ReID training contains two main source-training stages:

1. supervised multi-source pretraining;
2. feedback-regulated intermediate-domain training using MS-IDG, PMR, and DFC.

---

## Domain-Generalization Experiments

For leave-one-domain-out evaluation, use multiple labeled source domains and exclude the target domain from training.

Representative configurations are:

```text
Market-1501 + DukeMTMC-ReID → MSMT17
Market-1501 + MSMT17 → DukeMTMC-ReID
DukeMTMC-ReID + MSMT17 → Market-1501
Market-1501 + DukeMTMC-ReID + MSMT17 → CUHK03
```

Select the required source-target configuration and run:

```bash
python examples/train_aida.py
```

Evaluate the trained checkpoint:

```bash
python examples/test.py
```

---

## Source-Free Adaptation

SF-MIDA follows a two-stage protocol.

### Stage 1: Multi-Source Pretraining

The target dataset is excluded from labeled source training.

Examples:

```text
DukeMTMC-ReID + MSMT17 → Market-1501
Market-1501 + MSMT17 → DukeMTMC-ReID
Market-1501 + DukeMTMC-ReID → MSMT17
```

### Stage 2: Target-Only Refinement

Only the pretrained parameters and unlabeled target images are used. Source-domain images are not accessed.

Target pseudo-labels are regenerated at the beginning of every epoch using DBSCAN on Jaccard distances.

Default clustering parameters used in the paper are:

```text
eps         = 0.6
min_samples = 4
k1          = 30
k2          = 6
```

Noise samples are excluded from optimization.

The source-free objective contains:

* pseudo-label identity classification;
* pseudo-label triplet loss;
* pseudo-mirror regularization;
* relational consistency;
* DFC-regulated loss weighting.

Run the source-free configuration through:

```bash
python examples/train_aida.py
```

The exact pretraining checkpoint and target-refinement configuration must be specified in the selected experiment configuration.

---

## DFC Configuration

The default DFC settings used in the paper are:

| Parameter                        |            Value | Purpose                                          |
| -------------------------------- | ---------------: | ------------------------------------------------ |
| Initial IDG weight               |              1.0 | Initial intermediate-domain supervision strength |
| Initial PMR weight               |              1.0 | Initial pseudo-mirror regularization strength    |
| Maximum adaptive weight          |             10.0 | Upper clipping bound                             |
| Gradient-feedback factor (\beta) |              0.1 | Controls gradient-variance feedback              |
| Entropy-feedback factor (\gamma) |              0.5 | Controls uncertainty feedback                    |
| Entropy normalization (E_{\max}) |             10.0 | Normalizes prediction entropy                    |
| Gradient window                  | 100 mini-batches | Estimates gradient-norm variance                 |
| Update frequency                 | Every mini-batch | Updates DFC at every iteration                   |
| Mixing normalization             |          Softmax | Enforces probability-simplex constraints         |
| Maximum feedback scale           |              1.5 | Bounds source-mixing score modulation            |

The implementation is located in:

```text
idm/utils/dfc.py
```

---

## Ablation Studies

Ablation configurations are available in:

```text
idm/configs/ablation_configs.py
```

The main configurations include:

```text
A: IDM++ baseline
B: Baseline + MS-IDG
C: Baseline + MS-IDG + PMR
D: Full AIDA-ReID with MS-IDG + PMR + DFC
```

PMR component studies evaluate:

```text
MS-IDG only
MS-IDG + identity consistency
MS-IDG + relational consistency
MS-IDG + full PMR
```

Run the selected ablation configuration using:

```bash
python examples/train_aida.py
```

Additional instructions are available in:

```text
idm/configs/README_ABLATION.md
```

---

## Evaluation

Evaluate a trained model using:

```bash
python examples/test.py
```

The repository supports:

* CMC Rank-1;
* CMC Rank-5;
* CMC Rank-10;
* mean Average Precision;
* normalized mutual information;
* silhouette score;
* extended statistical analysis.

Evaluation implementations are located in:

```text
idm/evaluators.py
idm/evaluation_metrics/
idm/utils/stats_tests.py
```

The optional Cython evaluator can accelerate ranking evaluation.

---

## Result Analysis

Analyze saved results using:

```bash
python examples/analyze_results.py
```

This utility can be used to aggregate runs, compare configurations, and summarize evaluation metrics.

---

## Feature Visualization

Generate feature-space visualizations using:

```bash
python examples/visualize_features.py
```

Visualization utilities are implemented in:

```text
idm/utils/visualization.py
```

They can be used for:

* t-SNE visualization;
* UMAP visualization, when enabled;
* embedding comparison;
* domain-distribution analysis;
* clustering-quality inspection.

---

## Main Results

### Multi-Source Domain Generalization

| Training Sources       | Unseen Target | Rank-1 |  mAP |
| ---------------------- | ------------- | -----: | ---: |
| Market + Duke          | MSMT17        |   57.8 | 39.9 |
| Market + MSMT17        | Duke          |   78.5 | 64.3 |
| Duke + MSMT17          | Market        |   89.7 | 76.8 |
| Market + Duke + MSMT17 | CUHK03        |   51.3 | 49.9 |

### Source-Free Adaptation

| Target        | Rank-1 |  mAP |
| ------------- | -----: | ---: |
| Market-1501   |   97.2 | 86.4 |
| DukeMTMC-ReID |   95.4 | 84.8 |
| MSMT17        |   69.1 | 59.5 |

### Single-Source UDA

| Transfer        | Rank-1 |  mAP |
| --------------- | -----: | ---: |
| Market → Duke   |   92.5 | 84.9 |
| Duke → Market   |   95.3 | 89.2 |
| Market → MSMT17 |   72.1 | 58.9 |
| Duke → MSMT17   |   73.5 | 46.1 |
| MSMT17 → Duke   |   86.7 | 75.9 |
| MSMT17 → Market |   96.8 | 89.6 |

Results may vary slightly across hardware, software versions, and random seeds.

---

## Reproducibility

For reproducible experiments:

1. use the same dataset splits;
2. use the provided experiment configuration;
3. preserve the reported random seeds;
4. use the same pretrained backbone;
5. retain the reported optimizer and scheduler settings;
6. use the same DBSCAN parameters for source-free refinement;
7. report the exact checkpoint used for evaluation.

The paper uses:

```text
Backbone: ImageNet-pretrained ResNet-50
Input size: 256 × 128
Optimizer: AdamW
Weight decay: 5 × 10^-4
Batch size: 64
UDA/DG learning rate: 3.5 × 10^-4
Source-free learning rate: 1.0 × 10^-4
UDA epochs: 60
DG epochs: 50
Source-free refinement epochs: 30
```

DG and source-free experiments use GeM pooling and Sharpness-Aware Minimization. Single-source UDA follows the IDM++ pipeline with domain-specific batch normalization and global average pooling.

---

## Citation

Please cite the paper when using this code or its components.

```bibtex
@article{iqbal2026aidareid,
  title   = {AIDA-ReID: Feedback-Regulated Intermediate-Domain Learning for Generalizable and Source-Free Person Re-Identification},
  author  = {Iqbal, Sundas and Tian, Qing and Ali, Danish and Gou, Jianping and Ou, Weihua},
  journal = {Pattern Recognition},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Update the volume, pages, DOI, and publication status after the final journal record becomes available.

---

---

## Dataset Notice

The datasets are not included in this repository. Their copyright and licensing conditions remain with the original owners.

Users are responsible for:

* obtaining authorized access;
* following academic-use restrictions;
* complying with dataset licenses;
* avoiding redistribution of protected data.

---

## License

This repository is intended for academic research use.

Before public release, add a `LICENSE` file that is compatible with the licenses of all upstream code used by this project. Dataset licenses are separate from the source-code license.

---

## Contact

For questions regarding the paper or implementation, contact:

* **Sundas Iqbal:** [sundasiqbal058@gmail.com](mailto:sundasiqbal058@gmail.com)
* **Qing Tian:** [tianqing@nuist.edu.cn](mailto:tianqing@nuist.edu.cn)
* **Danish Ali:** [danishalikhan545@gmail.com](mailto:danishalikhan545@gmail.com)

For bug reports and implementation questions, please open a GitHub issue and include:

* the command used;
* the selected configuration;
* software and CUDA versions;
* dataset and transfer setting;
* complete error log.

```
```
