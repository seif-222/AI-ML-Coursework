# AI and Machine Learning Coursework

I wrote these notebooks while working through Daniel Bourke’s PyTorch course, both parts of fast.ai, and Andrej Karpathy’s *Neural Networks: Zero to Hero*. The collection also includes my own experiments on Kaggle’s Spaceship Titanic dataset. It contains 46 notebooks, along with Python implementations, saved outputs, and selected checkpoints.

## Where to start

| Track | What I implemented | Open an example |
| --- | --- | --- |
| [PyTorch with Daniel Bourke](Zero_To_Mastery/) | Image classification, modular training code, experiment tracking, and a Gradio interface for food classification | [FoodVision model deployment notebook](Zero_To_Mastery/AI_course/model_deployment.ipynb) |
| [fast.ai Part 1](Fast_AI/Part_1/Notebooks/) | Decision trees for tabular data, text classification, and collaborative filtering | [Recommendation systems](Fast_AI/Part_1/Notebooks/recommendation_systems.ipynb) |
| [fast.ai Part 2](Fast_AI/Part_2/Notebooks/) | A callback-based learner, convolutional networks, and diffusion models. The notebooks progress from the training loop to sampling images. | [Building the learner](Fast_AI/Part_2/Notebooks/05_Learner.ipynb) · [Fashion-MNIST diffusion samples](Fast_AI/Part_2/Notebooks/18_Karras.ipynb) |
| [Neural Networks: Zero to Hero](Zero_To_Hero/) | Manual backpropagation, byte-pair encoding, a character-level GPT, and a GPT training pipeline | [Tokenization](Zero_To_Hero/Before_GPT/Tokenization.ipynb) · [GPT training script](Zero_To_Hero/NanoGPT/GPT_train.py) |
| [Spaceship Titanic](Kaggle_Learning/Spaceship_Titanic/) | Compared five tabular classifiers across preprocessing experiments using a fixed validation split. The notebook shows how the model rankings changed. | [Experiments and comparison plots](Kaggle_Learning/Spaceship_Titanic/spaceship_titanic_BASELINE_1.ipynb) · [Kaggle submission files](Kaggle_Learning/Spaceship_Titanic/Submissions/) |

## Run a small model on CPU

This example runs a forward pass through my [TinyVGG implementation](Zero_To_Mastery/going_modular/tinyVGG.py). It takes one 64 × 64 RGB image and produces three class scores. It uses randomly initialized weights and a synthetic image, so you can check the model wiring without downloading a dataset.

These commands create a Python 3.12 environment on Linux and install the CPU build of PyTorch.

```bash
git clone https://github.com/seif-222/AI-ML-Coursework.git
cd AI-ML-Coursework
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

```bash
python - <<'PY'
import torch
from Zero_To_Mastery.going_modular.tinyVGG import VGGmodel

model = VGGmodel(input_layers=3, hidden_layers=10, output_layers=3).eval()
with torch.inference_mode():
    logits = model(torch.zeros(1, 3, 64, 64))
print(tuple(logits.shape))
PY
```

Expected output is `(1, 3)`.

## Working with the notebooks

GitHub can display saved notebook outputs, including training curves, image samples, and model comparisons. To rerun an experiment, check the setup cells in that notebook. The coursework spans several environments.

- **Spaceship Titanic** includes the train and test CSVs under [data](Kaggle_Learning/Spaceship_Titanic/data/). Start the notebook from its own folder so its relative data path resolves.
- **fast.ai Part 2** uses the course’s `miniai` modules. Some notebooks refer to the original local or Kaggle `course22p2` path. Point those imports at your own course checkout and check for CUDA calls before running training on CPU.
- **FoodVision** includes the [Gradio app source](Zero_To_Mastery/demos/food_vision_big_food101/app.py). Before launching it, provide the class list, trained weights, and example images referenced in that file.

My separate [PicoChat project](https://github.com/seif-222/PicoChat) builds on this coursework with tokenizer training, pretraining, supervised fine-tuning, and a custom inference engine.
