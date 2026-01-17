# Fashion-MNIST Classification with Deep Learning 👗👟

## 📝 Overview
This project focuses on building and evaluating a Deep Learning model to classify images from the **Fashion-MNIST** dataset. Fashion-MNIST is a collection of Zalando article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

The primary goal was to implement a Multi-Layer Perceptron (MLP) and perform **hyperparameter tuning** to observe how model complexity affects classification accuracy.

## 🛠️ Key Components
* **Data Preprocessing:** Normalization of pixel values to the [0, 1] range.
* **Architecture Design:** Comparison of two different Multi-Layer Perceptron (MLP) structures.
* **Hyperparameter Tuning:** Adjusting neurons, layers, and adding regularization.
* **Evaluation:** Analysis using Accuracy curves and a Confusion Matrix.

---

## 🏗️ Model Architectures

To meet the project requirements for hyperparameter tuning, two versions of the model were developed and compared:

### Model A (Baseline)
- **Architecture:** 1 Hidden Layer (128 neurons, ReLU activation).
- **Training:** 15 Epochs.
- **Parameters:** ~101,000 trainable parameters.
- **Role:** Served as a benchmark for performance.

### Model B (Tuned)
- **Architecture:** 2 Hidden Layers (512 and 256 neurons, ReLU activation).
- **Regularization:** Integrated **Dropout (0.2)** layers to prevent overfitting.
- **Training:** 15 Epochs with a 20% validation split.
- **Role:** Designed to handle complex features and improve generalization.


---

## 📈 Performance Analysis

### Confusion Matrix Insights
A detailed analysis of the Confusion Matrix revealed:
1. **High Success:** The models nearly perfectly identify classes with distinct shapes, such as **Bags** and **Ankle Boots**.
2. **Common Errors:** The most frequent misclassifications occur between **Shirts** and **T-shirts/tops**. Due to the low resolution (28x28), these items share very similar edge features.



---

## 💡 Conclusions & Future Improvements
* **Data Normalization:** Proper scaling (dividing by 255.0) is critical. Without it, the model fails to converge, leading to random guessing (~10% accuracy).
* **Model Capacity:** Increasing the number of neurons improves training accuracy, but requires regularization (like Dropout) to maintain high test performance.
* **Next Steps:** To significantly improve the distinction between similar items (like shirts), the next step would be implementing a **Convolutional Neural Network (CNN)** to better capture spatial hierarchies in images.

---

## 💻 How to use
1. Open the `.ipynb` file in **Google Colab**.
2. Run the initialization cells to download the Fashion-MNIST dataset automatically via Keras.
3. Follow the cell-by-cell execution to reproduce the preprocessing, training, and evaluation steps.
