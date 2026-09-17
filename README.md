# Breast Cancer Classification using a Neural Network

## Project Overview

This project implements a Feedforward Artificial Neural Network (ANN) to
classify breast tumors as **Malignant (M)** or **Benign (B)** using the
Breast Cancer Wisconsin Diagnostic Dataset.

## Dataset

-   Samples: **569**
-   Features: **30 numerical features**
-   Target:
    -   **M** → Malignant
    -   **B** → Benign

Removed columns: - `id` - `Unnamed: 32`

## Data Preprocessing

-   Removed unnecessary columns
-   Encoded target labels (`M=1`, `B=0`)
-   Standardized features using `StandardScaler`
-   Split data:
    -   Train: **70%**
    -   Validation: **15%**
    -   Test: **15%**

## Neural Network Architecture

    Input (30)
       ↓
    Dense(16, ReLU)
       ↓
    Dense(8, ReLU)
       ↓
    Dense(1, Sigmoid)

## Hyperparameters

  Parameter       Value
  --------------- ----------------------
  Optimizer       Adam
  Learning Rate   0.001
  Loss            Binary Crossentropy
  Batch Size      32
  Epochs          100 (Early Stopping)
  Patience        10

## Results

### Neural Network

-   Test Accuracy: **97.67%**
-   Precision: **0.98**
-   Recall: **0.97**
-   F1-Score: **0.98**

### Logistic Regression Baseline

-   Accuracy: **97.67%**

## Loss Curve Analysis

Training and validation loss decreased consistently during the early
epochs. Around epoch 34, the validation loss stopped improving while the
training loss continued to decrease, indicating mild overfitting. Early
stopping restored the best model weights automatically.

## Experiment Table

  ------------------------------------------------------------------------------
  Experiment   Hidden Layers   Learning Rate   Batch Size   Dropout   Result
  ------------ --------------- --------------- ------------ --------- ----------
  1            16 → 8          0.001           32           No        Baseline

  2            32 → 16         0.001           32           No        Fill in

  3            64 → 32         0.001           32           No        Fill in

  4            16 → 8          0.01            32           No        Fill in

  5            16 → 8          0.0001          32           No        Fill in

  6            16 → 8          0.001           16           No        Fill in

  7            16 → 8          0.001           64           No        Fill in

  8            16 → 8          0.001           32           0.3       Fill in
  ------------------------------------------------------------------------------

## Libraries

-   TensorFlow / Keras
-   Scikit-learn
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn

## Run

``` bash
pip install -r requirements.txt
python model.py
```

## Conclusion

The ANN successfully classified breast cancer tumors with **97.67% test
accuracy**, matching the Logistic Regression baseline. Early stopping
reduced overfitting and produced a robust model suitable for this binary
classification task.
