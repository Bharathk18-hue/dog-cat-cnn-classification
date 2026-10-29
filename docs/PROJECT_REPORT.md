# Development of a Convolutional Neural Network (CNN) for Dog–Cat Image Classification Using Three Convolutional Layers

> **How to use this report:** Sections marked `[RESULT: ...]` must be filled in from your own run
> (`outputs/metrics.json`, `outputs/classification_report.txt`, and the PNG graphs). No experimental values
> have been invented. Replace the placeholders with your real numbers and insert the images from `outputs/`.
> Fill in your name, roll number, guide and institution on the title page.

**Student:** [Your name] **Roll No.:** [ ] **Guide:** [ ] **Institution:** [ ] **Date:** [ ]

---

## 1. Abstract
This project develops a Convolutional Neural Network with exactly three convolutional layers to classify images as dog or cat. Images from the Kaggle Dogs vs. Cats dataset are cleaned (corrupted files and exact duplicates removed), resized to 128 × 128 pixels, normalized to the 0–1 range and split 70/15/15 into training, validation and test sets using a seeded, stratified split. The network (Conv32–Conv64–Conv128 with max pooling, a 128-neuron dense layer, dropout and a sigmoid output) is trained with Adam and binary cross-entropy, using EarlyStopping and ModelCheckpoint. Performance is measured with test accuracy, loss, precision, recall, F1-score and a confusion matrix. A prediction function and a Streamlit web application demonstrate use on new images. The final test accuracy obtained was **[RESULT: test accuracy after running]**.

## 2. Introduction
Image classification is a core computer-vision task. Traditional approaches required hand-designed features, while deep learning, especially CNNs, learns features directly from pixels. Distinguishing dogs from cats is a standard benchmark because the two classes vary greatly in breed, pose, lighting and background, which makes it a good learning problem for beginners.

## 3. Problem Statement
Given an RGB photograph, automatically decide whether it contains a dog or a cat using a CNN that has exactly three convolutional layers, and evaluate the model honestly on images it has never seen.

## 4. Objectives
1. Prepare a clean, leakage-free dataset (resize, normalize, 70/15/15 split).
2. Build a CNN with exactly three Conv2D layers.
3. Train with validation, EarlyStopping and ModelCheckpoint.
4. Evaluate using accuracy, loss, precision, recall, F1, confusion matrix and learning curves.
5. Provide a prediction function, model saving/loading and a simple web interface.

## 5. Existing System and Limitations
* **Manual sorting/tagging:** slow, costly and inconsistent.
* **Classical machine learning (e.g., SVM on hand-crafted features such as HOG or colour histograms):** requires feature engineering and generally performs worse on varied natural photos.
* **Fully connected networks on raw pixels:** ignore spatial structure and need very many parameters.
* **Large pretrained networks (e.g., ResNet, VGG):** accurate but heavy and harder for a beginner to understand and explain.

## 6. Proposed System
A compact CNN with three convolutional blocks trained from scratch. Convolutional layers learn features automatically, pooling reduces size, dropout reduces overfitting, and a single sigmoid neuron outputs the probability of *Dog*. The system includes a reproducible data pipeline, full evaluation, a reusable prediction function and a Streamlit app.

## 7. Dataset Description
* **Source:** Kaggle Dogs vs. Cats (about 25,000 labelled JPEG images, 12,500 per class, varying sizes).
* **Subset used:** `[RESULT: images per class used, e.g. 2000]` (set by `PER_CLASS`).
* **Structure:** `dataset/cats/` (label 0) and `dataset/dogs/` (label 1).
* **Cleaning:** unreadable files skipped; exact duplicates removed by MD5 hash. `[RESULT: number skipped corrupted / duplicate, printed by the script]`
* **Split:** 70% train / 15% validation / 15% test, stratified, `random_state=42`. `[RESULT: train / validation / test counts]`
* **Preprocessing:** convert to RGB, resize to 128 × 128, divide by 255 (range 0–1).

## 8. System Requirements
| Item | Requirement |
|---|---|
| Language | Python 3.10–3.12 |
| Libraries | TensorFlow/Keras, NumPy, Matplotlib, Scikit-learn, Pillow, Streamlit |
| Environment | Google Colab (free T4 GPU) or VS Code / Jupyter |
| Hardware | Any modern PC (≥ 8 GB RAM); GPU recommended |

## 9. Methodology
1. Collect file paths and labels → 2. remove corrupted/duplicate files → 3. stratified seeded split of the file list → 4. load and resize images, normalize → 5. build CNN → 6. train with validation → 7. keep best model → 8. evaluate on the untouched test set → 9. predict new images → 10. save/reload and deploy with Streamlit.

Splitting file paths *before* loading images, and removing duplicates *before* splitting, prevents data leakage. A fixed seed (42) makes the split and training setup reproducible where practical (GPU operations can still introduce small run-to-run differences).

## 10. CNN Architecture
| # | Layer | Output shape | Parameters |
|---|---|---|---|
| – | Input | 128×128×3 | 0 |
| 1 | **Conv2D** 32 filters, 3×3, ReLU | 126×126×32 | 896 |
| – | MaxPooling2D (2×2) | 63×63×32 | 0 |
| 2 | **Conv2D** 64 filters, 3×3, ReLU | 61×61×64 | 18,496 |
| – | MaxPooling2D | 30×30×64 | 0 |
| 3 | **Conv2D** 128 filters, 3×3, ReLU | 28×28×128 | 73,856 |
| – | MaxPooling2D | 14×14×128 | 0 |
| – | Flatten | 25,088 | 0 |
| – | Dense 128, ReLU | 128 | 3,211,392 |
| – | Dropout (0.5) | 128 | 0 |
| – | Dense 1, sigmoid | 1 | 129 |

Total parameters: 3,304,769 (taken from the Keras model summary; the code asserts exactly three `Conv2D` layers).

**Convolutional layer 1 (32 filters):** scans the raw image for simple low-level patterns such as edges, corners and colour changes.
**Convolutional layer 2 (64 filters):** combines those into textures and small parts, such as fur patterns or the curve of an ear.
**Convolutional layer 3 (128 filters):** combines parts into larger, more class-specific structures such as a snout, a face or body shape.

**Label convention:** 0 = Cat, 1 = Dog. Output = P(Dog); > 0.5 → Dog.

**Compilation:** optimizer Adam, loss binary cross-entropy, metric accuracy.

## 11. Implementation Details
* `src/train_model.py` – complete pipeline (data preparation, model, training, evaluation, prediction, saving/loading).
* `notebooks/dog_cat_cnn.ipynb` – the same code split into notebook cells.
* `src/prepare_dataset.py` – sorts Kaggle images into `dataset/cats` and `dataset/dogs`.
* `app.py` – Streamlit interface (upload → preview, class, confidence).
* Training callbacks: **EarlyStopping** (monitor `val_loss`, patience 3, restore best weights) and **ModelCheckpoint** (save best `val_loss` model to `dog_cat_cnn.keras`).
* Prediction uses exactly the same loading/resizing/normalization as training and handles invalid paths and unsupported files.

## 12. Experimental Setup
| Parameter | Value |
|---|---|
| Image size | 128 × 128 × 3 |
| Batch size | 32 |
| Maximum epochs | 20 (EarlyStopping, patience 3) |
| Optimizer / loss | Adam (default learning rate) / binary cross-entropy |
| Dropout rate | 0.5 |
| Random seed | 42 |
| Platform | `[RESULT: e.g. Google Colab T4 GPU]` |
| Epochs actually run | `[RESULT: from metrics.json]` |
| Training time | `[RESULT: optional]` |

## 13. Results and Evaluation Metrics
`[RESULT: fill from outputs/metrics.json and outputs/classification_report.txt]`

| Metric | Value |
|---|---|
| Test loss | [RESULT] |
| Test accuracy | [RESULT] |
| Precision (Dog = positive class) | [RESULT] |
| Recall (Dog = positive class) | [RESULT] |
| F1-score (Dog = positive class) | [RESULT] |

Paste the full classification report here: `[RESULT]`

## 14. Confusion Matrix and Graphs
* Confusion matrix: `outputs/confusion_matrix.png` `[RESULT: insert image and interpret: how many cats predicted as dogs and vice versa]`
* Accuracy curves: `outputs/accuracy_graph.png` `[RESULT: insert image]`
* Loss curves: `outputs/loss_graph.png` `[RESULT: insert image]`
* Interpretation guide: if training accuracy is much higher than validation accuracy, the model is overfitting; if both are low, it is underfitting. `[RESULT: write your own observation after seeing the graphs]`

## 15. Advantages and Limitations
**Advantages:** simple and easy to explain; trains from scratch quickly; automatic feature learning; reproducible, leakage-aware pipeline; deployable through a web interface.
**Limitations:** only two classes (no "neither" option, so any image gets a dog/cat label); limited data and a shallow network cap accuracy; no data augmentation; hash-based deduplication cannot find near-duplicate images; performance may drop on unusual poses, heavy occlusion or low-quality photos; trained from scratch rather than using transfer learning.

## 16. Future Enhancements
Data augmentation (flips, rotations, zoom); batch normalization; learning-rate scheduling; transfer learning with MobileNet or ResNet; training on the full 25,000 images; adding an "other/unknown" class; Grad-CAM visualisation to show what the model looks at; mobile or cloud deployment.

## 17. Conclusion
A CNN with exactly three convolutional layers was designed to classify dog and cat images, with a leakage-aware data pipeline, full evaluation and a deployable interface. `[RESULT: add 2–3 sentences summarizing your actual test accuracy/F1 and what you observed.]`

## 18. References
1. Y. LeCun, Y. Bengio, G. Hinton, "Deep learning," *Nature*, vol. 521, pp. 436–444, 2015.
2. Y. LeCun et al., "Gradient-based learning applied to document recognition," *Proc. IEEE*, vol. 86, no. 11, 1998.
3. D. P. Kingma, J. Ba, "Adam: A method for stochastic optimization," arXiv:1412.6980, 2014.
4. N. Srivastava et al., "Dropout: A simple way to prevent neural networks from overfitting," *JMLR*, vol. 15, 2014.
5. Kaggle, "Dogs vs. Cats" dataset. https://www.kaggle.com/c/dogs-vs-cats
6. TensorFlow/Keras documentation. https://www.tensorflow.org/api_docs
7. Scikit-learn documentation (metrics). https://scikit-learn.org/stable/modules/model_evaluation.html
8. I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.
