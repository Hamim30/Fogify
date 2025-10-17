# 🌫️ SyntheticFogGenerator

**SyntheticFogGenerator** is a lightweight, dependency-minimal Python toolkit for generating **synthetic foggy images** using **OpenCV** and **NumPy** — no PyTorch or TensorFlow required.  
It provides efficient fog simulation through **simplified** and **hybrid depth estimation** methods, suitable for research in *image dehazing, dataset augmentation, and visibility analysis.*

---

## 🚀 Features

- 🧠 **Two depth estimation modes**
  - *Simple:* Fast edge-gradient-based approximation  
  - *Hybrid:* Combines edge, brightness, and positional priors for realistic fog
- ⚙️ **No deep learning frameworks needed** (OpenCV + NumPy only)
- 📦 **Batch dataset processing** (Main → Dataset)
- 🎚️ **Adjustable fog density** via β values (2–7 recommended)
- 🖼️ **Visualization support** for fog and depth maps
- 💻 **Platform-independent**, minimal dependencies

---

## 🧩 Installation

```bash
pip install opencv-python numpy matplotlib
