# CNN — FashionMNIST in PyTorch

A walkthrough of building image classifiers in PyTorch on the FashionMNIST dataset, going from a simple linear baseline to a convolutional neural network modeled after Tiny VGG (from the [CNN Explainer](https://poloclub.github.io/cnn-explainer/) project).

The notebook follows the standard deep-learning workflow end to end: prepare data → build model → train → evaluate → iterate.

## What's inside

`CNN.ipynb` builds and compares three models on FashionMNIST (28×28 grayscale images, 10 clothing classes):

| Model | Architecture | Device |
|---|---|---|
| Model 0 | Linear baseline (Flatten → Linear → ReLU → Linear) | CPU |
| Model 1 | Same architecture as Model 0 | GPU |
| Model 2 | CNN — two Conv blocks (Conv2d → ReLU → Conv2d → ReLU → MaxPool2d) + classifier head | GPU |

Each model is trained with `CrossEntropyLoss` and SGD (`lr=0.1`), then evaluated on the test set using a shared `eval_model` function that returns loss and accuracy.

## Topics covered

- Loading datasets with `torchvision.datasets`
- Batching with `DataLoader`
- Writing reusable `train_step` / `test_step` / `eval_model` functions
- Device-agnostic code (`cuda` if available, else `cpu`)
- Timing experiments with `timeit`
- Stepping through `nn.Conv2d` and `nn.MaxPool2d` to understand shape transformations
- Comparing model performance across architectures and devices

## Requirements

```
torch
torchvision
matplotlib
tqdm
requests
torchmetrics
```

A GPU is recommended for Models 1 and 2 but not required — everything will run on CPU, just more slowly.

## Running the notebook

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
jupyter notebook CNN.ipynb
```

Or open it directly in [Google Colab](https://colab.research.google.com/) — `Open notebook` → `GitHub` tab → paste the repo URL.

The notebook downloads `helper_functions.py` from the [Learn PyTorch repo](https://github.com/mrdbourke/pytorch-deep-learning) at runtime for the `accuracy_fn` utility, and FashionMNIST itself is downloaded to a local `data/` folder on first run.

## Dataset

[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) — 60,000 training and 10,000 test grayscale images across 10 classes (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

## Troubleshooting

**"The 'state' key is missing from 'metadata.widgets'" when viewing on GitHub.** This is a known Colab bug — Colab saves tqdm progress bar widget state in a way that doesn't match the Jupyter widget spec, and GitHub's renderer rejects it. The fix is to strip the broken metadata block before pushing:

```bash
python -c "import json,sys; nb=json.load(open(sys.argv[1])); nb['metadata'].pop('widgets',None); json.dump(nb, open(sys.argv[1],'w'), indent=1)" CNN.ipynb
```

Alternatively, in Colab use **Edit → Clear all outputs** before downloading. The widgets metadata only stored the visual state of completed progress bars, so removing it doesn't affect anything you'd want to keep.

## Credits

The CNN architecture and overall structure follow Daniel Bourke's [Learn PyTorch for Deep Learning](https://www.learnpytorch.io/) course. The Tiny VGG reference comes from the [CNN Explainer](https://poloclub.github.io/cnn-explainer/) visualization by Polo Club at Georgia Tech.

