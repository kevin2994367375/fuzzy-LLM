# Fuzzy-LLM: An Framework for Bearing Fault Diagnosis

Our work introduces FuzzyLLM, a novel framework that bridges the gap between the numerical world of sensor data and the semantic reasoning space of Large Language Models (LLMs). By constructing fuzzy knowledge—including semantic feature descriptions and uncertainty-aware soft labels—we enhance the diagnostic accuracy, interpretability, and robustness of LLMs for industrial fault diagnosis tasks.

![FuzzyLLM Framework](path/to/your/framework_diagram.png) 
*Figure 1: The framework of FuzzyLLM.*

---


## 🔧 Installation

It is highly recommended to use a Conda environment to manage dependencies.

**1. Clone the repository:**
```bash
git clone https://github.com/kevin2994367375/fuzzy-LLM.git
cd FuzzyLLM
```
**2. Create and activate the Conda environment:**
```bash
conda create -n fuzzyllm python=3.10
conda activate fuzzyllm
```
**3. Install PyTorch with GPU support:**
Visit the official PyTorch website to get the correct installation command for your specific CUDA version. For example:
```bash
# Example for CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
**4. Install all other dependencies:**
```bash
pip install -r requirements.txt
```
**5. Download Models (Optional but Recommended):**
To avoid repeated downloads, you can pre-download the language models you intend to use and place them in a ./models/ directory. For gated models like Llama-3 or Gemma, you must first request access on their respective Hugging Face pages and log in via the terminal:
Generated bash
huggingface-cli login
Use code with caution.
```bash
📂 Project Structure
Generated code
.
├── data/
│   ├── raw/              # Place raw CWRU, PU, HUST datasets here
│   ├── processed/        # Directory for intermediate data splits
├── models/               # (Optional) Store pre-downloaded LLM models
├── outputs/              # All experimental results, logs, and figures are saved here
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── fuzzy_encoder.py
│   └── models/
│       └── model.py
├── main.py                     # Main script for running a single experiment
└── README.md                   # This file
Use code with caution.
```
