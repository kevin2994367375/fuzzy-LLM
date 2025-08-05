# Fuzzy-LLM: An Interpretable Framework for Bearing Fault Diagnosis

This repository contains the official implementation for the paper: **"FuzzyLLM: A Fuzzy Knowledge Enhanced Large Language Model for Interpretable Bearing Fault Diagnosis"**.

Our work introduces FuzzyLLM, a novel framework that bridges the gap between the numerical world of sensor data and the semantic reasoning space of Large Language Models (LLMs). By constructing fuzzy knowledge—including semantic feature descriptions and uncertainty-aware soft labels—we enhance the diagnostic accuracy, interpretability, and robustness of LLMs for industrial fault diagnosis tasks.

![FuzzyLLM Framework](path/to/your/framework_diagram.png) 
*Figure 1: The overall three-stage framework of FuzzyLLM.*

---

## 🚀 Features

- **Novel Fuzzy Knowledge Construction**: Implements an adaptive fuzzy encoder and a KNN-based soft label generator to translate numerical data into an LLM-native format.
- **State-of-the-Art LLM Fine-tuning**: Utilizes modern LLMs (e.g., Qwen2.5, Llama-3) and Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA/QLoRA for efficient training.
- **Comprehensive Ablation Studies**: Includes automated scripts to systematically evaluate the contribution of each component (fuzzy input, soft labels) and the model's robustness to noise.
- **Automated Experiment Pipelines**: Provides powerful Python scripts (`run_full_ablation.py`, `run_model_comparison.py`) to run complex experiment suites with a single command.
- **Publication-Ready Visualization**: Includes scripts to generate high-quality, professional charts and graphs for research papers.

---

## 🔧 Installation

It is highly recommended to use a Conda environment to manage dependencies.

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/FuzzyLLM.git
cd FuzzyLLM
Use code with caution.
Markdown
**2. Create and activate the Conda environment:**
Generated bash
conda create -n fuzzyllm python=3.10
conda activate fuzzyllm
Use code with caution.
Bash
**3. Install PyTorch with GPU support:**
Visit the official PyTorch website to get the correct installation command for your specific CUDA version. For example:
Generated bash
# Example for CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
Use code with caution.
Bash
**4. Install all other dependencies:**
Generated bash
pip install -r requirements.txt
Use code with caution.
Bash
**5. Download Models (Optional but Recommended):**
To avoid repeated downloads, you can pre-download the language models you intend to use and place them in a ./models/ directory. For gated models like Llama-3 or Gemma, you must first request access on their respective Hugging Face pages and log in via the terminal:
Generated bash
huggingface-cli login
Use code with caution.
Bash
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

python3 run_alpha_ablation.py
Use code with caution.
Bash
Results: A summary file will be generated in outputs/alpha_ablation/. You can then use plot_alpha_sensitivity_combined.py to visualize the results.
