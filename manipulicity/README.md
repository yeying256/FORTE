# Manipulicity — VLM-Based Physical Mass Estimation

This branch explores using **Vision-Language Models (VLMs)** to infer the
physical parameters of objects (starting with **mass**) from visual input.
The estimated mass will later feed into an impedance controller for compliant
robot interaction.

---

## What it does

Given a single RGB image of an object that is **grasped** or **hanging** from
a robot end-effector, `mass_estimator.py` queries a VLM and returns:

| Field | Description |
|---|---|
| `mass_kg` | Best-estimate mass (float, kg) |
| `mass_kg_range` | Plausible `[lower, upper]` range (kg) |
| `material_guess` | Inferred material (e.g. "metal", "plastic") |
| `object_description` | Brief description of the object |
| `confidence` | `"low"` / `"medium"` / `"high"` |
| `reasoning` | Chain-of-thought explanation |

The estimator uses a **two-stage Chain-of-Thought prompt**: the model first
reasons step-by-step (object identity, material, dimensions, density
calculation, sanity check, uncertainty), then emits a structured JSON result.

---

## Supported VLM backends

| Backend | Model examples | How to select |
|---|---|---|
| **OpenAI** | `gpt-4o-mini` (default), `gpt-4o` | Default; set `OPENAI_API_KEY` |
| **Google Gemini** | `gemini-1.5-flash`, `gemini-1.5-pro` | Model name starts with `gemini` |
| **Local (Ollama)** | `llama3.2-vision:11b` (default local) | Prefix model with `ollama/` |

---

## Installation

```bash
pip install -r requirements.txt
```

Set the relevant API key as an environment variable:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Gemini
export GEMINI_API_KEY="AIza..."

# Local (Ollama) — no key needed; install Ollama and pull the model
ollama pull llama3.2-vision:11b
```

---

## Usage

### Command line — single image

```bash
# OpenAI (default)
python mass_estimator.py --image path/to/image.jpg

# Gemini
python mass_estimator.py --image path/to/image.jpg \
    --model gemini-1.5-flash --gemini_api_key AIza...

# Local Ollama
python mass_estimator.py --image path/to/image.jpg \
    --model ollama/llama3.2-vision:11b

# Optional flags
#   --api_key       sk-...    OpenAI key (or OPENAI_API_KEY env var)
#   --gemini_api_key AIza...  Gemini key (or GEMINI_API_KEY env var)
#   --local_base_url http://localhost:11434/v1   custom local server URL
```

### Python API

```python
from mass_estimator import MassEstimator

# OpenAI
estimator = MassEstimator(api_key="sk-...")

# Gemini
estimator = MassEstimator(gemini_api_key="AIza...", model="gemini-1.5-flash")

# Local Ollama
estimator = MassEstimator(model="ollama/llama3.2-vision:11b")

# Accept file path, PIL.Image, or numpy ndarray
result = estimator.estimate("gripper_holding_mug.jpg")

print(result["mass_kg"])        # e.g. 0.35
print(result["confidence"])     # e.g. "medium"
print(result["reasoning"])
```

---

## Batch testing

Use `test_mass_estimator.py` to run the estimator on every image in a folder
and get a summary report.

### 1. Add test images

Place your images under `tests/images/` (subdirectories are supported):

```
tests/
└── images/
    ├── apple.jpg
    ├── mug.png
    └── metal_block.jpg
```

### 2. Run the test runner

```bash
# OpenAI
python test_mass_estimator.py --test_dir tests/images --api_key sk-...

# Gemini
python test_mass_estimator.py --test_dir tests/images \
    --model gemini-1.5-flash --gemini_api_key AIza...

# Local Ollama
python test_mass_estimator.py --test_dir tests/images \
    --model ollama/llama3.2-vision:11b
```

### 3. Optional flags

| Flag | Description |
|---|---|
| `--save_json results.json` | Save all results to a JSON file |
| `--ground_truth tests/gt.json` | Compare predictions against known masses (computes MAE) |
| `--stop_on_error` | Abort on the first failed image |

### 4. Ground-truth file format

Create `tests/gt.json` to enable error metrics:

```json
{
  "apple.jpg": 0.182,
  "mug.png": 0.350,
  "metal_block.jpg": 1.200
}
```

---

## Project structure

```
manipulicity/
├── mass_estimator.py        # Core VLM mass-estimation module + CLI
├── test_mass_estimator.py   # Batch test runner
├── tests/
│   └── images/              # Place test images here
├── requirements.txt         # Python dependencies
├── 方法.md                  # Detailed method description (Chinese)
└── README.md                # This file
```

---

## Notes

- `gpt-4o-mini` provides a good cost/accuracy balance for most objects.
  Switch to `--model gpt-4o` for higher accuracy on ambiguous scenes.
- The estimated mass is a *semantic* estimate based on visual appearance and
  world knowledge; it is not a measurement.  For impedance control the value
  is used as a soft prior that can be refined online.
- Local Ollama inference requires sufficient VRAM: `llama3.2-vision:11b` needs
  ~8 GB; use `llama3.2-vision:90b` for maximum accuracy (~50 GB VRAM).
