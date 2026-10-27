# Egyptian Currency Image Classification

A notebook-based computer vision project for classifying images of Egyptian banknotes by denomination and, where represented in the labels, note series (for example, `10` and `10 (new)`). The repository explores image enhancement, transfer learning with EfficientNet, and CNN classifiers trained on SIFT- and HOG-based visual representations.

> **Scope:** The notebooks perform image-class classification. They do not verify that a banknote is genuine, detect counterfeit notes, or establish monetary value from a live transaction. Do not use these experimental models for financial decisions or as a substitute for a validated currency-authentication system.

## Project contents

| File or folder | Purpose |
|---|---|
| `enhanced_money_detection_with_visuals.ipynb` | Image enhancement, class balancing, EfficientNetB3 fine-tuning, and test evaluation. |
| `improved_money_detection.ipynb` | Earlier EfficientNetB0 transfer-learning experiment. |
| `filters.ipynb` | SIFT, HOG, and combined SIFT+HOG image transformations with CNN training and evaluation. |
| `cam.ipynb`, `test.ipynb` | Inference experiments that load the combined SIFT/HOG model. |
| `combined_sift_hog_model.h5`, `sift_model.h5`, `hog_model.h5` | Saved Keras models from the filter experiments. |
| `efficientnetb0_egyptian_currency_model.h5` | Saved EfficientNetB0 model artifact. |
| `Egyptian new currency/` | Currency dataset used by the notebooks. |
| `Egyptian currency/`, `Euro coins and bills/` | Additional image collections included in the repository. |

The notebooks are research and demonstration workflows; the repository does not currently provide a packaged application or a single command-line prediction interface.

## Classes

The enhanced EfficientNetB3 notebook uses nine classes:

```text
1, 5, 10, 10 (new), 20, 20 (new), 50, 100, 200
```

These are the folder names used as labels by that notebook. Confirm the classes in your training directory before training or running inference; the class set and ordering must match the model and label encoder used for predictions.

## Requirements

- Python with Jupyter Notebook or JupyterLab
- TensorFlow / Keras
- OpenCV (`opencv-python`)
- NumPy, Pillow, scikit-learn, scikit-image, pandas, matplotlib, seaborn, and tqdm

Create and activate a virtual environment, then install the notebook dependencies:

```bash
python -m venv .venv
```

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow opencv-python numpy pillow scikit-learn scikit-image pandas matplotlib seaborn tqdm jupyter
```

TensorFlow installation depends on your operating system and hardware. Follow the [official TensorFlow installation instructions](https://www.tensorflow.org/install) if the command above is not suitable for your environment. Training EfficientNet models can take several minutes or longer and benefits from a compatible GPU.

## Dataset layout

The training notebooks expect a split dataset with one subfolder per class:

```text
Egyptian new currency/
└── dataset/
    ├── train/
    │   ├── 1/
    │   ├── 5/
    │   ├── 10/
    │   ├── 10 (new)/
    │   ├── 20/
    │   ├── 20 (new)/
    │   ├── 50/
    │   ├── 100/
    │   └── 200/
    ├── valid/
    │   └── <same class folders>
    └── test/
        └── <same class folders>
```

Each image's containing folder is its label. The validation and test splits must use the same class folders as training. The notebooks contain machine-specific absolute paths such as `D:\Fake money detection\Egyptian new currency\dataset\train`; update `train_dir`, `val_dir`, and `test_dir` to point to your local dataset before running the cells.

For example, with the notebook opened from the repository root:

```python
from pathlib import Path

project_dir = Path.cwd()
dataset_dir = project_dir / "Egyptian new currency" / "dataset"
train_dir = dataset_dir / "train"
val_dir = dataset_dir / "valid"
test_dir = dataset_dir / "test"
```

If the dataset is stored elsewhere, set `project_dir` to its parent directory. The repository also contains standalone image collections; those are not a replacement for the `train` / `valid` / `test` class-folder structure used by the training notebooks.

## Run the experiments

1. Install the dependencies and open the repository in Jupyter.
2. Choose a notebook:
   - `enhanced_money_detection_with_visuals.ipynb` for the EfficientNetB3 experiment.
   - `improved_money_detection.ipynb` for the EfficientNetB0 experiment.
   - `filters.ipynb` to compare SIFT, HOG, and combined SIFT/HOG preprocessing.
3. Update all dataset paths in the notebook to match your local folders.
4. Run the cells in order. Training notebooks create models in the notebook's working directory when their save cells run.

The enhanced EfficientNetB3 pipeline resizes images to 224 × 224, applies colored-image denoising and CLAHE contrast enhancement, uses image augmentation for training, and fine-tunes the last layers of an ImageNet-pretrained EfficientNetB3. It trains with class weights and early stopping.

The filter experiment resizes each image to 224 × 224 and creates SIFT keypoint, HOG visualization, or combined SIFT+HOG images. An EfficientNetB0-based CNN then classifies those transformed images. These are image transformations used as model inputs; the saved files are Keras classifiers, not standalone SIFT/HOG descriptor matchers.

## Reported notebook results

The checked-in `filters.ipynb` outputs the following test accuracy values on its 290-image test split:

| Experiment | Reported test accuracy |
|---|---:|
| SIFT-transformed images | 99.31% |
| HOG-transformed images | 95.52% |
| Combined SIFT+HOG images | 99.31% |

The `enhanced_money_detection_with_visuals.ipynb` output reports 86.90% test accuracy on 290 images for its EfficientNetB3 experiment. The difference in results reflects different experiment setups; these figures are notebook-reported results, not guarantees of performance on independently collected notes, different cameras, lighting, wear, or backgrounds.

## Loading a saved model

Keras artifacts can be loaded with TensorFlow, for example:

```python
from tensorflow.keras.models import load_model

model = load_model("combined_sift_hog_model.h5")
```

A usable prediction pipeline also needs the exact preprocessing function and class-to-index mapping used during training. The included `cam.ipynb` and `test.ipynb` refer to a `label_encoder.pkl` file that is not present at the repository root, and they contain absolute Windows paths. Run the training notebook to recreate the model and label encoder, or update the inference notebook to use a matching encoder and local model path before relying on predictions. Do not infer class labels from a model's output index without that mapping.

## Limitations

- Dataset paths are hard-coded to a Windows machine and must be updated.
- The notebooks are not a unified application and do not expose a stable prediction API.
- Saved models require matching preprocessing and class-label mappings.
- Image augmentation and train/validation/test split quality affect the reported metrics; the notebook outputs alone do not establish performance on real-world currency images.
- The enhanced notebook includes horizontal flips. Since banknote designs can be orientation-dependent, validate this augmentation choice against the intended use case.
- The model classifies image appearance; it does not authenticate a note or detect counterfeits.

## License and data

No license file is included in the repository. Standard copyright restrictions apply to reuse of the code unless the owner adds a license. Image and dataset rights may have separate terms; check them before redistributing images or deploying a model trained on them.
