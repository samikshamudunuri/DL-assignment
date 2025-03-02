# DL-assignment
# CIFAR-10 Classification using a Feedforward Neural Network

## **Requirements**
- Ensure **Python 3.x** is installed.
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```
- Run the model training with:
  ```bash
  python train.py
  ```
- Evaluate the model:
  ```bash
  python evaluate.py
  ```

## **Code Structure**
- `train.py`: Script for training the feedforward neural network.
- `evaluate.py`: Script for testing the trained model and generating a confusion matrix.
- `models.py`: Defines the architecture of the neural network.
- `utils.py`: Contains helper functions for data loading and preprocessing.
- `requirements.txt`: Lists required Python packages.
- `report.pdf`: Contains a summary of the results and findings.

## **Dataset**
- The **CIFAR-10 dataset** is automatically downloaded using `torchvision.datasets`.
- It is preprocessed by normalizing pixel values and split into **training**, **validation**, and **test** sets.

## **Model Architecture**
The feedforward neural network consists of:
- **Input layer**: 3072 neurons (Flattened 32x32x3 images)
- **Hidden layers**: Two fully connected layers with 256 and 128 neurons.
- **Dropout layer**: Applied after the first hidden layer (dropout probability = 0.3) to reduce overfitting.
- **Output layer**: 10 neurons (corresponding to the 10 CIFAR-10 classes) without softmax, as it is handled in the loss function.

## **Hyperparameter Tuning**
The model supports flexible hyperparameter selection, including:
- **Hidden Layers**: 2
- **Neurons per Layer**: [256, 128]
- **Optimizer**: Adam
- **Learning Rate**: 1e-3
- **Batch Size**: 32
- **Activation Function**: ReLU
- **Regularization**: Dropout (0.3 probability)

## **Training & Evaluation**
- Training is performed over **5 epochs** using the **Adam optimizer** and **Cross-Entropy Loss**.
- The model is trained on the CIFAR-10 training set and evaluated using the validation and test sets.
- Accuracy is computed based on correct predictions over total samples.
- A **confusion matrix** is plotted to visualize classification performance.

## **Findings from the Experiments**
1. **Optimal Hyperparameters:**
   - The best performance was achieved using **two hidden layers**, **ReLU activation**, and the **Adam optimizer**.
   - **Batch size of 32** provided a good balance between speed and performance.
2. **Effect of Dropout:**
   - A **dropout probability of 0.3** helped reduce overfitting without significantly degrading performance.
3. **Accuracy Performance:**
   - **Validation Accuracy:** ~70%
   - **Test Accuracy:** ~69%

## **Final Thoughts**
- The chosen hyperparameters provided **good generalization** while preventing overfitting.
- **ReLU activation** improved training speed compared to Sigmoid.
- The **Adam optimizer** performed better than basic SGD in this setting.

## **Evaluation Results**
- The final trained model achieves **~70% accuracy on validation data**.
- The **confusion matrix** shows that the model performs well but has difficulty distinguishing visually similar classes.
- The training and evaluation pipeline is **reproducible and easy to modify** for further experimentation.

