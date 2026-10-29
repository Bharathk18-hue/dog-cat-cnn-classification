# Dog vs Cat Image Classification using a 3-Convolutional-Layer CNN

A beginner-friendly CNN (TensorFlow/Keras) that classifies images as **Cat (label 0)** or **Dog (label 1)**.
The sigmoid output is the probability of **Dog**: above 0.5 means Dog, otherwise Cat.

## Project status (please read)
The code was smoke-tested on small synthetic images (pipeline, 3-Conv2D check, training, evaluation,
saving/loading, prediction, notebook execution, Streamlit app loading). **It has NOT been trained on the real
Kaggle dataset.** `dog_cat_cnn.keras` and all accuracy numbers do not exist yet. You create them by running the
steps below. Nothing in this project contains made-up results.

## Folder structure
```
dog_cat_classification/
├── dataset/
│   ├── cats/                 <- put cat images here
│   └── dogs/                 <- put dog images here
├── notebooks/
│   └── dog_cat_cnn.ipynb     <- same pipeline as train_model.py, cell by cell (Colab/Jupyter)
├── src/
│   ├── train_model.py        <- loads data, trains, evaluates, saves model, demo predictions
│   └── prepare_dataset.py    <- optional helper: sorts Kaggle images into dataset/cats and dataset/dogs
├── docs/
│   ├── PROJECT_REPORT.md     <- academic report (results marked [RESULT: ...] until you fill them in)
│   └── VIVA_QA.md            <- 15 viva questions with answers
├── app.py                    <- Streamlit web app (upload image -> Dog/Cat + confidence)
├── requirements.txt          <- Python packages
├── dog_cat_cnn.keras         <- CREATED by training (best model)
├── outputs/                  <- CREATED by training (graphs, metrics.json, report, model summary)
└── README.md
```

## 1. Get the dataset
Use the Kaggle **Dogs vs. Cats** dataset (25,000 images). Download it from kaggle.com (search "dogs vs cats"),
unzip it, then run:
```bash
python src/prepare_dataset.py path/to/unzipped_folder 2000
```
This copies 2000 images per class into `dataset/cats` and `dataset/dogs`. (Or copy files yourself;
cat files should be in `cats/`, dog files in `dogs/`.)

| Subset | Images per class | Use |
|---|---|---|
| Quick test | 500 | check the code runs (minutes) |
| Project run (recommended) | 2000 | real results |
| Full | up to 12500 | best accuracy, slow |

## 2. Run it

### Option A: Google Colab (easiest, free GPU)
1. Upload the `dog_cat_classification` folder to Google Drive (with the dataset inside).
2. Open `notebooks/dog_cat_cnn.ipynb` in Colab. Runtime -> Change runtime type -> T4 GPU.
3. Mount Drive in a first cell and `%cd` into the `notebooks` folder:
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   %cd /content/drive/MyDrive/dog_cat_classification/notebooks
   ```
4. Run all cells (Runtime -> Run all).

### Option B: VS Code / terminal
```bash
cd dog_cat_classification
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python src/train_model.py
```
Quick test with 500 images per class (instead of 2000):
* VS Code terminal (PowerShell, the default): `$env:PER_CLASS=500` then `python src/train_model.py`
* Command Prompt (cmd): `set PER_CLASS=500` then `python src/train_model.py`
* Mac/Linux: `PER_CLASS=500 python src/train_model.py`

The setting lasts only for that terminal window. Use `$env:PER_CLASS=2000` (or close the terminal) to go back to the full run.
Use Python 3.10-3.12. If TensorFlow fails to install, use Colab.

## 3. What you get after training
In `outputs/`: `sample_images.png`, `accuracy_graph.png`, `loss_graph.png`, `confusion_matrix.png`,
`metrics.json` (test loss, accuracy, precision, recall, F1), `classification_report.txt`, `model_summary.txt`,
and sample prediction images. The model is saved as `dog_cat_cnn.keras`.
All values come from the trained model and test predictions. Precision/recall/F1 treat **Dog as the positive class**
(the classification report shows both classes).

## 4. Run the web app
After `dog_cat_cnn.keras` exists:
```bash
streamlit run app.py
```
A browser tab opens. Upload a JPG/PNG to see the preview, predicted class and confidence.

## 5. Predict with code
`train_model.py` defines `predict_image(path)`, which handles missing/unsupported files without crashing.
It is demonstrated on one test cat, one test dog, a bad path, and again with the reloaded model.

## How data leakage is avoided
Corrupted files and exact duplicates (MD5 hash) are removed **before** a stratified, seeded (42) 70/15/15
split of the file list; the code asserts the splits do not overlap. Note: near-duplicates (same photo
re-saved/cropped) cannot be detected by hashing.

## Troubleshooting
* `Folder not found: dataset/cats`: dataset not set up (step 1).
* `Not enough images found`: the folders are empty or contain unsupported files (only jpg/jpeg/png are used).
* Out of memory: lower `PER_CLASS`.
* Training slow: use Colab GPU or a smaller `PER_CLASS`.
* Any other error: copy the full message and ask for help.
