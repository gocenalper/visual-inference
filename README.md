# 🚀 Visual Inference - Deepfake Detection

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![Docker](https://img.shields.io/badge/Docker-Supported-brightgreen.svg)

🔍 **Visual Inference** is a deepfake detection system that utilizes **Xception-based feature extraction** and **Transformer-based sequence modeling** to classify **real vs. fake images**. This project provides a **Dockerized solution** to simplify model deployment and inference.

---

## 📌 Features

✔️ **Xception-based** feature extraction for frame analysis  
✔️ **Transformer Encoder** for sequence modeling  
✔️ **Cross-Attention Mechanism** to refine embeddings  
✔️ **Docker support** for easy deployment  
✔️ **Inference statistics** to analyze model performance  

---

## 🔧 Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone git@github.com:gocenalper/visual-inference.git
cd visual-inference
```

### **2️⃣ Install Dependencies (For Local Usage)**
```bash
pip install torch torchvision timm tqdm pillow
```

---

## 🐳 Running with Docker

We provide a **lightweight Docker image** for running the model without manually installing dependencies.

### **1️⃣ Build the Docker Image**
```bash
docker build -t dfdc-inference .
```

### **2️⃣ Run the Docker Container (Mounting Dataset & Code)**
Run the following command to **mount your dataset and code inside the container**:
```bash
docker run --rm -it -v "$(pwd)":/app dfdc-inference
```

### **3️⃣ Run with GPU Support (Optional)**
If your machine has **CUDA-enabled GPUs**, use:
```bash
docker run --gpus all --rm -it -v "$(pwd)":/app dfdc-inference
```

---

## 🖼 Dataset Structure

The dataset should be **mounted** in the following format:

```
/DFDC/
    ├── REAL/
    │   ├── TRAIN/
    │   │   ├── video_0001/
    │   │   │   ├── frame_01.jpg
    │   │   │   ├── frame_02.jpg
    │   │   ├── video_0002/
    │   ├── TEST/
    │   ├── VAL/
    ├── FAKE/
    │   ├── TRAIN/
    │   │   ├── video_0003/
    │   │   │   ├── frame_01.jpg
    │   │   │   ├── frame_02.jpg
    │   ├── TEST/
    │   ├── VAL/
```

---

## 🚀 Running Inference

Once the **Docker container is running**, the model will process **test images** and print real-time statistics:

```
🔍 Running Untrained Model Inference on Test Data...

📌 Image 1: Predicted = FAKE, Actual = REAL, ❌ Incorrect
📌 Image 2: Predicted = REAL, Actual = FAKE, ❌ Incorrect
📌 Image 3: Predicted = REAL, Actual = REAL, ✅ Correct
📌 Image 4: Predicted = FAKE, Actual = FAKE, ✅ Correct

📊 **Inference Statistics (Before Training)**
🔹 Total Images Processed: 1000
🟢 Real Predictions: 500 (50.0%)
🔴 Fake Predictions: 500 (50.0%)
✅ Correct Predictions: 495 (49.5%) Accuracy
```

---

## 🛠 Troubleshooting

### **1️⃣ Docker Build Fails: "No space left on device"**
Run
