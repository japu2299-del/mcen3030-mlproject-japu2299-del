# Bean Type Classification Using Machine Learning and Random Forest
**Course:** MCEN 3030  
**Author:** Jake Purpura  
**Dataset:** "Dry Bean." UCI Machine Learning Repository, 2020, https://doi.org/10.24432/C50S4B.

---

## Table of Contents
1. [Problem Description & Initial Prompt](#1-problem-description--initial-prompt)
2. [Background: Random Forest](#2-background-random-forest)
3. [Code 1: Baseline Model](#3-code-1-baseline-model)
4. [Code 2: Tuned Model & Hyperparameter Discussion](#4-code-2-tuned-model--hyperparameter-discussion)
5. [Feature Importance](#5-feature-importance)

---

## 1. Problem Description & Initial Prompt

The goal of this project is to build a machine learning model that can predict the **type of dry bean** from geometric measurements. The dataset contains **13,611 samples**, each described by **16 continuous geometric features** (such as area, perimeter, major axis length, roundness, etc.) and labeled with one of **7 bean types**: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, and SIRA.

This is a **multiclass classification problem** — given a set of shape measurements, the model must decide which of the 7 bean types a sample belongs to.

### Initial Prompt Given to LLM

> "I would like to do a machine learning project where the goal is to predict the type of bean from geometric measurements. I have a CSV file dataset where there are 16 different variables with 7 beans that are the target guesses. Can we talk about the best way to model this dataset and approach the project? There are several suggestions/approaches we should take including random forest and neural networks and including a confusion matrix to help see where the model is going wrong."

The LLM outlined a full pipeline: data preprocessing, model selection, train/test splitting, confusion matrix evaluation, and iteration. It suggested **Random Forest as the primary model**, with SVM and Neural Networks as comparison points. The reasoning was that Random Forest is naturally suited to multiclass tabular data, requires no feature scaling, handles correlated features well, and provides built-in feature importance rankings — all of which are directly relevant to this dataset.

---

## 2. Background: Random Forest

### Why Random Forest?

Random Forest was chosen as the modeling approach for several reasons specific to this problem:

- The 16 geometric features are likely **correlated** (e.g., area and perimeter scale together), and Random Forest handles correlated features gracefully by randomly subsampling features at each split
- The target variable has **7 classes**, and Random Forest handles multiclass problems natively through majority voting across trees
- No **feature scaling** is required, unlike neural networks or SVMs
- The model produces **feature importance scores**, which gives interpretable insight into which geometric measurements matter most for classification
- It is robust against overfitting because the randomness built into each tree acts as a natural regularizer

### How Random Forest Works

A single **decision tree** splits data by asking threshold questions about features until a classification is reached. However, a single tree tends to overfit — it memorizes the training data and performs poorly on new data.

A **Random Forest** addresses this by building hundreds of independent trees and combining their votes. The class receiving the most votes wins. Two sources of randomness are injected deliberately:

1. **Bootstrap Sampling (Bagging):** Each tree is trained on a different random sample of the data, drawn with replacement. Some samples appear multiple times; others not at all. This forces diversity between trees.

2. **Random Feature Subsets:** At every split point within every tree, only a random subset of features (default: √16 = 4) is considered as a candidate for splitting. This prevents all trees from making identical splits and ensures they are decorrelated.

Because each tree makes different errors, those errors cancel out across the ensemble — the combined prediction is more stable and accurate than any individual tree.

As the LLM explained:

> "Many imperfect models combined outperform any single perfect-looking model."

### Confusion Matrix

A confusion matrix is used to evaluate multiclass performance beyond simple accuracy. The diagonal shows correct predictions per class. Off-diagonal entries reveal *which* classes are being confused with each other — providing both diagnostic and scientific insight into which bean types are geometrically similar.

---

## 3. Code 1: Baseline Model

The baseline model was implemented in MATLAB using the Statistics and Machine Learning Toolbox. The dataset was loaded from the `.xlsx` file and split 70/30 into training and test sets using stratified sampling to preserve class proportions.

**See code:** [`code_1/bean_classification.m`](code_1/bean_classification.m)

### Baseline Hyperparameters

| Parameter | Value |
|---|---|
| Number of Trees | 100 |
| MinLeafSize | 1 |
| NumPredictorsToSample | √16 = 4 (default) |
| Train/Test Split | 70% / 30% stratified |

### Baseline Results

**Overall Test Accuracy: 92.75%**

| Bean Type | Precision | Recall | F1 Score |
|---|---|---|---|
| BARBUNYA | 0.945 | 0.907 | 0.925 |
| BOMBAY | 1.000 | 1.000 | 1.000 |
| CALI | 0.920 | 0.941 | 0.930 |
| DERMASON | 0.914 | 0.934 | 0.924 |
| HOROZ | 0.965 | 0.941 | 0.953 |
| SEKER | 0.962 | 0.946 | 0.954 |
| SIRA | 0.876 | 0.882 | 0.879 |

### Baseline Confusion Matrix

![Baseline Confusion Matrix](images/confusion_matrix_baseline.png)
<!-- Save your confusion matrix screenshot as images/confusion_matrix_baseline.png -->

### Discussion

The baseline model achieved strong performance at 92.75%. Notable observations:

- **BOMBAY** achieved a perfect 1.000 across all metrics, indicating it is geometrically very distinct from all other bean types
- **SIRA** had the weakest performance (F1 = 0.879), with 75 actual SIRA samples being misclassified as DERMASON — the largest single source of error in the entire model
- **BARBUNYA** had the second-lowest recall at 90.7%, with misclassifications spreading toward both CALI (24 samples) and SIRA (12 samples), suggesting BARBUNYA sits geometrically between these two classes in feature space
- The OOB error plot showed the model stabilizing at approximately 40 trees, meaning anything beyond ~50 trees provides no additional accuracy benefit

---

## 4. Code 2: Tuned Model & Hyperparameter Discussion

### What Are Hyperparameters?

A **hyperparameter** is a setting that controls how the learning process works — it is chosen before training and never adjusted by the model itself. In contrast, a regular model parameter (like which feature to split on) is learned from the data.

For Random Forest, the key hyperparameters are:

- **Number of Trees:** More trees increases stability but with diminishing returns. The OOB error plot showed convergence around 40–50 trees for this dataset.
- **NumPredictorsToSample:** The number of features randomly considered at each node split. Default is √p = 4. Lower values increase tree diversity; higher values make individual splits stronger but trees more correlated.
- **MinLeafSize:** Controls tree depth by requiring a minimum number of samples at each leaf. Smaller values allow deeper trees (more memorization); larger values produce shallower, more generalized trees.

### Tuning Process

Bayesian optimization was run using `fitcensemble` with `OptimizeHyperparameters = 'auto'` over 30 trials. The optimizer explored three ensemble methods (Bag, AdaBoostM2, RUSBoost) and converged on:

| Parameter | Optimized Value |
|---|---|
| Method | Bag (Random Forest) |
| NumLearningCycles | 497 |
| MinLeafSize | 2 |

This confirmed that **Bag (Random Forest) is the correct method** for this dataset. The optimizer's best result was 92.63% — marginally below the original 92.75%, confirming the baseline was already near-optimal. The key actionable finding was that MinLeafSize = 2 (slightly shallower trees) was preferred over the default of 1.

A `NumPredictorsToSample` sweep was also run across values {3, 4, 6, 8} to identify the best feature sampling rate.

**See code:** [`code_2/bean_classification_tuned.m`](code_2/bean_classification_tuned.m)

### Tuned Hyperparameters

| Parameter | Baseline | Tuned |
|---|---|---|
| Number of Trees | 100 | 50 |
| MinLeafSize | 1 | 2 |
| NumPredictorsToSample | 4 (default) | [best from sweep] |

### Tuned Results

**Overall Test Accuracy: [your tuned accuracy here]%**

| Bean Type | Precision | Recall | F1 Score |
|---|---|---|---|
| BARBUNYA | | | |
| BOMBAY | | | |
| CALI | | | |
| DERMASON | | | |
| HOROZ | | | |
| SEKER | | | |
| SIRA | | | |

### Tuned Confusion Matrix

![Tuned Confusion Matrix](images/confusion_matrix_tuned.png)
<!-- Save your tuned confusion matrix screenshot as images/confusion_matrix_tuned.png -->

### What Changed Between Models

[Write 2-3 sentences here in your own words comparing the two confusion matrices. For example: did SIRA/DERMASON confusion improve? Did BARBUNYA recall go up? Did reducing tree count to 50 maintain accuracy?]

---

## 5. Feature Importance

The Random Forest model tracks which features contribute most to correct classifications by measuring how much accuracy drops when each feature is randomly permuted (OOB permutation importance).

### Feature Importance Plot

![Feature Importance](images/feature_importance.png)
<!-- Save your feature importance bar chart as images/feature_importance.png -->

### Discussion

[Write 3-5 sentences here in your own words. Some things to address:
- Which features ranked highest? (e.g., Area, Perimeter, MajorAxisLength?)
- Which features were near zero and contributed little?
- Does the ranking make intuitive sense for distinguishing bean shapes?
- Does this connect to where the model struggled — for example, do SIRA and DERMASON score similarly on the top-ranked features, explaining their confusion?]

---

## Repository Structure

```
/
├── README.md
├── images/
│   ├── confusion_matrix_baseline.png
│   ├── confusion_matrix_tuned.png
│   └── feature_importance.png
├── code_1/
│   └── bean_classification.m
└── code_2/
    └── bean_classification_tuned.m
```
