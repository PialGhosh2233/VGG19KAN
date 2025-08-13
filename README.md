# VGG19-KAN: Hybrid CNN with Kolmogorov-Arnold Network Layers

This repository implements a hybrid deep learning model that combines a **pre-trained VGG19** convolutional backbone with **Kolmogorov-Arnold Network (KAN) linear layers** for flexible and powerful representation learning on image classification tasks.

---

## 🚀 Project Overview

The model leverages:

* **VGG19**: A pre-trained CNN for extracting high-level image features.
* **KANLinear layers**: Novel fully connected layers inspired by the Kolmogorov-Arnold representation, providing improved non-linear mapping and expressivity.
* **Data augmentation**: Techniques like random flips, rotations, color jitter, and random erasing for robust training.

The architecture is designed to replace traditional fully connected layers in standard CNNs with KANLinear layers for enhanced performance.

---

## ⚙️ Requirements

Python 3.x and the following packages:

```bash
torch
torchvision
scikit-learn
matplotlib
numpy
```

Install via pip:

```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

---

## 🖥️ Model Architecture

**VGG19KAN**:

1. **Feature Extractor**: Pre-trained VGG19 convolutional layers (`vgg19.features`)
2. **Adaptive Average Pooling**: Reduce feature maps to fixed size `(7x7)`
3. **KANLinear Layers**:

   * `kan1`: 25088 → 512
   * `kan2`: 512 → 1024
   * `kan3`: 1024 → `num_classes`

The model is trained with **CrossEntropyLoss** and optimized using **AdamW**.

---

## 🔄 Training & Evaluation

The training loop includes:

* Training with backpropagation
* Validation phase for monitoring overfitting
* Test evaluation with **accuracy, precision, recall, and F1-score**

Metrics are tracked per epoch and can be visualized using `matplotlib`.

```python
train_accuracies, val_accuracies, test_accuracies, train_losses, val_losses, test_losses, test_precisions, test_recalls, test_f1_scores = run(
    model, criterion, optimizer, train_loader, val_loader, test_loader
)
```

---

## 🔧 How to Use

1. Prepare your dataset with **training, validation, and test splits** in folders.
2. Update dataset paths in the code:

```python
train_dataset = datasets.ImageFolder(root='PATH_TO_TRAIN', transform=train_transform)
val_dataset = datasets.ImageFolder(root='PATH_TO_VAL', transform=val_transform)
test_dataset = datasets.ImageFolder(root='PATH_TO_TEST', transform=test_transform)
```

3. Run training:

```bash
python train_vgg19_kan.py
```

4. Save the trained model:

```python
torch.save(model.state_dict(), "vgg19_kan.pth")
```

---

## 🔬 References

* [VGG19 PyTorch Implementation](https://pytorch.org/vision/stable/models.html)
* [Efficient-KAN: Kolmogorov-Arnold Network](https://github.com/Blealtan/efficient-kan)
* [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)

---

## 📌 Notes

* GPU is recommended for training.
* KANLinear layers replace standard fully connected layers for better representation learning.
* Data augmentation improves model robustness and generalization.

Do you want me to do that next?
