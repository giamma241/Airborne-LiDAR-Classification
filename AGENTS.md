# Agents.md Guide for OpenAI Codex — Airborne LiDAR Transformer-Based Framework

This `AGENTS.md` file provides instructions and conventions for OpenAI Codex or other AI agents contributing to this point cloud classification framework based on Hydra, PyTorch Lightning, and transformer-based models.

---

## Updated Project Structure

```plaintext
.
├── configs/               # Hydra-based configuration system
│   ├── model/             # Model-specific hyperparameters
│   ├── data/              # Dataset and data loader settings
│   ├── trainer/           # Training setup (epochs, GPUs, callbacks)
│   ├── paths/             # Input/output path references
│   └── train.yaml         # Main entry config for training
│
├── src/                   # Main source code
│   ├── data/              # Dataset loading, augmentation, preprocessing
│   ├── models/            # Deep learning architectures (PointNet++, SuperPoint, etc.)
│   ├── utils/             # Logging, transforms, evaluation helpers
│   ├── train.py           # Training entrypoint (Hydra + Lightning CLI)
│   └── eval.py            # Evaluation entrypoint
│
├── notebooks/             # Jupyter notebooks for experimentation
├── scripts/               # Bash scripts (job submission, data preparation)
├── tests/                 # Unit tests (pytest-based)
├── logs/                  # Hydra logs and Lightning experiment outputs
├── data/                  # External `.las` files and cached `.npz` blocks (excluded from Git)
```

---

## Codex Coding Guidelines

### Language & Libraries

* **Python 3.9+**, PyTorch, PyTorch Lightning, Hydra
* Use `numpy`, `torch`, `laspy`, `pdal` where relevant
* Avoid web-based visualizations; use `Open3D`, `matplotlib`, or `seaborn` instead

### Style & Format

* Follow **PEP8** and **Black** (`black src/`)
* Add **typed function definitions** and **NumPy-style docstrings**
* Use **`logging`** instead of `print`
* Code should be **modular**, **reproducible**, and **configurable**

### Configuration (Hydra)

* Use `configs/train.yaml`, `configs/eval.yaml` as entry points
* When introducing new models or datasets, create corresponding:

  * `configs/model/your_model.yaml`
  * `configs/data/your_dataset.yaml`
* Use `@hydra.main()` or `instantiate()` patterns when extending

---

## Tasks OpenAI Codex Can Help With

| Area          | Task Examples                                                   |
| ------------- | --------------------------------------------------------------- |
| `src/models/` | Implement or adapt transformer models (e.g., KPConv, PointMLP)  |
| `src/data/`   | Add support for `.las` or `.npz` datasets with preprocessing    |
| `src/utils/`  | Add evaluation metrics (e.g., mIoU, F1), logging, visualization |
| `configs/`    | Generate new model or training configs                          |
| `tests/`      | Add unit tests using `pytest` and mock datasets                 |
| `notebooks/`  | Create exploratory analysis or prediction visualizations        |

---

## Testing & Linting

Before any merge, Codex should ensure all checks pass:

```bash
# Unit tests
pytest tests/

# Linting
flake8 src/ --max-line-length=100

# Type checking
mypy src/

# Formatting
black src/
```

---

## Pull Request Checklist for Codex

When OpenAI Codex opens a PR, ensure:

1. A **clear description** of the purpose and scope
2. All new code is **typed and documented**
3. Hydra config files are updated if necessary
4. **Tests** exist or are extended for the feature
5. Code adheres to lint, type, and formatting standards

---

## Project-Specific Notes

* Models operate on **cached `.npz` blocks**, generated from `.las` via preprocessing
* Ground truth must be **spatially aligned** before training/eval
* Training is managed by **Hydra + PyTorch Lightning**
* Avoid direct `.las` reads during training: use `PointCloudDataset` loaders

---

By following this guide, OpenAI Codex can confidently assist in extending and maintaining the **Airborne LiDAR Transformer Framework**.
