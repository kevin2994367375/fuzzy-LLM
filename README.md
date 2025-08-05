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
**6. Command-Line Arguments & Configuration:**
Of course. Here is a comprehensive guide to all command-line arguments in `main.py`, written in English and formatted in Markdown, perfectly suited for your `README.md`.

The `main.py` script is designed to be highly configurable via command-line arguments. This allows for precise control over experiments without modifying the source code.

Below is a detailed explanation of each available argument.

### Core Control Parameters

These arguments define the fundamental setup and strategy for each experiment.

| Argument | Type | Default | Choices | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--dataset` | str | `cwru` | `cwru`, `pu`, `hust` | Specifies which dataset to use for the experiment. |
| `--strategy` | str | `fuzzy` | `numeric`, `fuzzy` | Sets the **input strategy**: `numeric` for scaled numerical text, `fuzzy` for semantic fuzzy text. |
| `--use_soft_labels`| flag | `False` | N/A | If set, enables the **output strategy** of using KNN-generated soft labels and the hybrid loss function for training. |

**Example:**
```bash
# Run an experiment on the PU dataset for the location task, using fuzzy inputs and soft labels.
python3 main.py --dataset pu --criterion location --strategy fuzzy --use_soft_labels
```

### Model & Fine-tuning Parameters

These arguments control the language model architecture and the specifics of the fine-tuning process.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model_name` | str | `./models/Qwen2.5-3B-Instruct`| Path to a local model or a Hugging Face Hub model ID. |
| `--tuning_method`| str | `lora` | The PEFT method to use. `lora` for 8-bit quantization, `qlora` for 4-bit. |
| `--lora_r` | int | `16` | The rank (`r`) of the LoRA adapter matrices. |
| `--lora_alpha` | int | `32` | The scaling factor for LoRA. Typically set to `2 * r`. |
| `--lora_dropout`| float| `0.1` | The dropout probability for the LoRA layers. |

### Training Hyperparameters

These arguments control the behavior of the training loop.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--epochs` | int | `3` | The total number of training epochs. |
| `--batch_size` | int | `8` | The number of samples per training batch. Adjust based on GPU memory. |
| `--learning_rate`| float| `2e-5` | The initial learning rate for the AdamW optimizer. |
| `--max_length` | int | `512` | The maximum token sequence length. Longer sequences will be truncated. |

### Fuzzy Knowledge & Loss Parameters

These are special parameters for the Fuzzy-LLM framework.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--fuzzy_config_name`| str | `quantile_based_5_levels`| The fuzzy encoding strategy to use from `config.yaml`. |
| `--knn_k` | int | `10` | The number of neighbors (`k`) for the KNN soft label generator. |
| `--soft_label_temperature`| float| `0.3` | The temperature `T` for sharpening soft labels (`T<1` sharpens, `T>1` smooths).|
| `--loss_alpha` | float| `0.5` | The weight `α` for the KL-divergence component in the hybrid loss. `0.0` for CE only, `1.0` for KL only. |

### Miscellaneous Parameters

These arguments control file paths, data handling, and debugging features.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--config` | str | `config.yaml` | Path to the main YAML configuration file. |
| `--data_dir` | str | `data/raw` | The root directory for the raw dataset files. |
| `--output_dir` | str | `outputs` | The root directory where all experimental results will be saved. |
