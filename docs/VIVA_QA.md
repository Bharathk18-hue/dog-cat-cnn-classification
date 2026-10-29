# Viva Questions and Answers (Beginner Friendly)

**1. What is a CNN?**
A Convolutional Neural Network is a neural network designed for images. It uses small sliding filters to detect patterns (edges, shapes, textures) instead of looking at every pixel independently.

**2. Why use a CNN for image classification?**
Images have spatial structure (nearby pixels are related). CNNs exploit this, use far fewer parameters than fully connected networks (filters are shared across the image), and detect a pattern wherever it appears.

**3. What is a convolutional layer?**
A layer that slides filters over the image and computes dot products, producing feature maps that show where each pattern was found. In this project: Conv1 (32 filters) finds edges/colours, Conv2 (64) finds textures and parts, Conv3 (128) finds larger shapes such as ears or snouts.

**4. Why exactly three convolutional layers?**
It was the project requirement. Three layers give a simple progression from low-level to higher-level features, are enough for a reasonable baseline, are fast to train on a free Colab GPU, and are easy to explain. More layers could be more accurate but are slower and more prone to overfitting on small data.

**5. What are filters and kernels?**
A kernel is the small grid of weights (here 3x3) that slides over the image. A filter is a kernel applied across all input channels; each layer has many filters (32, 64, 128), each learning a different pattern. The values are learned during training.

**6. What is ReLU?**
Rectified Linear Unit: f(x) = max(0, x). It passes positive values and turns negatives to zero, adding non-linearity so the network can learn complex patterns, and it is cheap to compute.

**7. What is max pooling?**
It splits a feature map into small windows (2x2) and keeps only the maximum of each. This halves width and height, reduces computation, and makes the model less sensitive to small shifts.

**8. What is flattening?**
Converting the final 3-D feature maps into one long 1-D vector so it can be fed to dense layers.

**9. What is a dense layer?**
A fully connected layer where every neuron connects to every input. Here a 128-neuron dense layer combines the extracted features, and a final 1-neuron dense layer produces the output.

**10. What is dropout?**
During training, randomly switching off a fraction of neurons (50% here). This stops neurons from relying on each other too much and reduces overfitting. Dropout is disabled at prediction time.

**11. What is sigmoid activation?**
A function that squashes any number into 0-1: 1/(1+e^-x). In this project the output is P(Dog); above 0.5 means Dog, otherwise Cat.

**12. What is binary cross-entropy?**
The loss function for two-class problems. It heavily penalizes confident wrong predictions and is small when the predicted probability is close to the true label (0 or 1).

**13. What is the Adam optimizer?**
An optimizer that updates the weights to reduce the loss. It combines momentum with a separate adaptive learning rate per weight, which usually converges quickly with little tuning.

**14. What are accuracy, precision, recall and F1-score?**
* Accuracy = correct predictions / all predictions.
* Precision = of images predicted as Dog, how many really are dogs (TP / (TP+FP)).
* Recall = of all real dogs, how many were found (TP / (TP+FN)).
* F1 = harmonic mean of precision and recall (2PR/(P+R)).

**15. What is overfitting and how can it be reduced?**
Overfitting is when the model memorizes training data and performs poorly on new data (training accuracy high, validation accuracy low). Reduce it with dropout, EarlyStopping, more data, data augmentation, a smaller model, and by keeping a separate validation/test set to detect it.

## Extra questions your examiner may ask
* *Why split into train/validation/test?* Train to learn, validation to tune/stop, test for an unbiased final score.
* *Why divide pixels by 255?* To scale values to 0-1, which makes training faster and more stable.
* *Why remove duplicates before splitting?* So identical images cannot appear in both training and test sets (data leakage would inflate accuracy).
* *What does EarlyStopping do?* Stops training when validation loss stops improving and restores the best weights.
