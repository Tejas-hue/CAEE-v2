# INSTALLATION GUIDE — CONTEXT AWARE EMPATHY ENGINE v2

> Everything you need to get up and running with the empathy engine.

---

## 1. Clone the Repository

```bash
git clone https://github.com/Tejas-hue/CAEE-v2.git
cd CAEE-v2
```

---

## 2. Create a Virtual Environment

**Using `venv` (recommended):**

```bash
python -m venv caee-env
```

**Activate it:**

- **Mac/Linux:**
  ```bash
  source caee-env/bin/activate
  ```
- **Windows:**
  ```bash
  caee-env\Scripts\activate
  ```

**Or using `conda`:**

```bash
conda create -n caee python=3.10
conda activate caee
```

---

## 3. Install Git LFS (Important)

> The model is over 700MB and tracked using Git LFS (Large File Storage).

If you haven’t installed Git LFS yet:

```bash
git lfs install
```

---

## 4. Install Python Requirements

```bash
pip install -r requirements.txt
```

---

## 5. Run the Model

```bash
python run_caee.py
```

> You’ll be prompted to enter a sentence.

> The model will return the predicted **psychological needs** associated with that sentence.

---

## 6. Folder Structure (v2)

```
CAEE-v2/
├── deberta_needs/                 # Model files + label encoder
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── spm.model
│   └── label_encoder.pkl
│
├── run_caee.py                    # Main script for prediction
├── requirements.txt               # All Python dependencies
├── INSTALL.md                     # This file
├── LICENSE                        # Open-source license
├── README.md                      # Project overview
```
