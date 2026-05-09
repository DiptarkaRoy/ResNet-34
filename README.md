# ResNet-34 CIFAR-10 Classifier

A modular deep learning project implementing ResNet-34 architecture to classify the CIFAR-10 dataset. This project is optimized for Apple Silicon (M3) using the MPS backend and managed with the `uv` package manager.

## 🚀 Features
* **Modular Architecture:** Separate modules for model definition, data loading, and training.
* **MPS Optimized:** Specific memory management (`empty_cache`) for efficient training on Mac.
* **Dynamic Configuration:** Command-line arguments for epochs, batch size, and image resolution.
* **Advanced Scheduling:** Implementation of Cosine Annealing learning rate decay.
* **Visual Tracking:** Integrated TensorBoard support for real-time loss and accuracy plots.
* **Scalable Hardware Support:** Automatically detects multiple GPUs and utilizes DataParallel for distributed training, while falling back to MPS (Metal Performance Shaders) on Mac or CPU when necessary.

---
## 📁 Project Structure
\`\`\`text
resnet34_project/
├── src/
│   ├── __init__.py      # Package exports
│   ├── model.py         # ResNet-34 & Residual Block architecture
│   ├── dataset.py       # CIFAR-10 pipeline & transformations
│   ├── train.py         # Training loop with argparse
│   └── predict.py       # Inference script for new images
├── models/              # Saved .pth weight files
├── data/                # Downloaded CIFAR-10 dataset
├── runs/                # TensorBoard event logs
└── pyproject.toml       # Environment & dependencies
\`\`\`

---

## 🛠️ Setup & Installation

Ensure you have [uv](https://github.com/astral-sh/uv) installed.

1. **Clone the repository:**
   \`\`\`bash
   git clone <your-repo-url>
   cd resnet34_project
   \`\`\`

2. **Sync dependencies:**
   \`\`\`bash
   uv sync
   \`\`\`

---

## 🏋️ Training

Run the training module from the root directory as a module. You can customize the run using flags:

\`\`\`bash
# Standard run (128px resolution)
uv run python -m src.train --epochs 50 --size 128

# Fast test run
uv run python -m src.train --epochs 5 --size 32 --batch_size 256
\`\`\`

### Monitoring Progress
Launch TensorBoard to view live training metrics:
\`\`\`bash
uv run tensorboard --logdir=runs
\`\`\`

---

### Note on Multi-GPU:
If you move this project from your MacBook M3 to a multi-GPU server (e.g., a Linux machine with 4x RTX 4090s), the script will automatically detect all available CUDA devices and distribute the batch across them.

## 🔮 Inference

To predict the class of a local image, use the \`predict.py\` script:

\`\`\`bash
uv run python -m src.predict --image_path "path/to/image.jpg"
\`\`\`

---

## 📊 Hyperparameters
| Parameter | Value |
| :--- | :--- |
| **Optimizer** | SGD (Momentum 0.9) |
| **Learning Rate** | 0.1 (Initial) |
| **Scheduler** | CosineAnnealingLR |
| **Weight Decay** | 5e-4 |
| **Loss Function** | CrossEntropyLoss |

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.