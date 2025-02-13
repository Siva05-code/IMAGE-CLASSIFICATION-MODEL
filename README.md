# IMAGE-CLASSIFICATION-MODEL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SIVAKARTHICK B

*INTERN ID*: : CT08FYO

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEEKS

*MENTOR*: NEELA SANTOSH

**MNIST Handwritten Digit Classification Using CNN**

### **Introduction**
Handwritten digit classification is a fundamental problem in computer vision and pattern recognition. In this project, we implemented a **Convolutional Neural Network (CNN)** model to classify handwritten digits from the **MNIST dataset**. The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), with 60,000 images used for training and 10,000 for testing. This project aims to achieve high accuracy in recognizing digits using deep learning techniques.

### **Tools and Technologies Used**
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow and Keras
- **Data Processing**: NumPy, Matplotlib
- **Dataset**: MNIST (Modified National Institute of Standards and Technology)
- **Libraries**:
  - `tensorflow.keras`: For building and training the CNN model
  - `numpy`: For numerical operations
  - `matplotlib.pyplot`: For data visualization
  - `os` and `random`: For handling and shuffling image files

### **Methodology**

#### **1. Data Preprocessing**
- Loaded the MNIST dataset from TensorFlow's Keras module.
- Normalized pixel values from the range **(0-255) to (0-1)** for better convergence.
- Reshaped the images to **(28, 28, 1)** to match the CNN input format.

#### **2. CNN Model Architecture**
The CNN model was designed with the following layers:
- **Convolutional Layer 1**: 32 filters, **3×3 kernel**, ReLU activation
- **MaxPooling Layer 1**: Pool size (2×2)
- **Convolutional Layer 2**: 64 filters, **3×3 kernel**, ReLU activation
- **MaxPooling Layer 2**: Pool size (2×2)
- **Convolutional Layer 3**: 64 filters, **3×3 kernel**, ReLU activation
- **Flatten Layer**: Converts the feature maps into a 1D vector
- **Fully Connected (Dense) Layer**: 64 neurons, ReLU activation
- **Output Layer**: 10 neurons (for digits 0-9), Softmax activation

#### **3. Model Compilation & Training**
- Used **Adam optimizer** for efficient gradient descent optimization.
- Used **Sparse Categorical Cross-Entropy** as the loss function.
- Trained the model for **10 epochs**, using **training data (80%)** and validating it on **test data (20%)**.

#### **4. Model Evaluation**
- The trained model achieved a high **test accuracy** of over **98%**.
- The performance was evaluated using **accuracy score** and **validation loss/accuracy trends**.
- A visualization of the accuracy trend was plotted to observe the learning curve.

### **Handwritten Digit Prediction on New Images**
- Loaded test images from a folder containing handwritten digit images.
- Preprocessed each image:
  - Converted to **grayscale**.
  - Resized to **28×28 pixels**.
  - Normalized pixel values.
  - Reshaped the image for CNN input.
- Passed the images through the trained model to predict digit labels.
- Displayed the images along with their predicted values in a grid format.

### **Applications of Handwritten Digit Recognition**
1. **Banking & Finance**: Automated check processing and signature verification.
2. **Postal Services**: Automated sorting of postal addresses based on handwritten zip codes.
3. **Education Sector**: Digital grading of handwritten answer scripts.
4. **Assistive Technology**: Helping visually impaired individuals convert handwritten text to speech.
5. **Forensic Analysis**: Handwriting recognition for fraud detection and criminal investigations.

### **Conclusion**
This project successfully implemented a **Convolutional Neural Network (CNN)** for recognizing handwritten digits with high accuracy. By leveraging deep learning techniques and efficient data preprocessing, we built a robust model that can classify digits from both the MNIST dataset and custom handwritten images. The application of this model extends to various real-world domains where handwritten text recognition is essential.


### **Output**


<img width="1186" alt="Image" src="https://github.com/user-attachments/assets/0fc18612-b8a3-447a-86e9-2e9fb0f7d247" />
