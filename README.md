# **Diabetes Prediction Machine Learning Model**  

## Description: 
This project focuses on developing a machine learning model to predict diabetes medication requirements using patient data. Built with Python, the project leverages libraries like Pandas, NumPy, Scikit-learn, and TensorFlow/Keras. It demonstrates data preprocessing, feature engineering, and the implementation of a Multilayer Perceptron (MLP) neural network.  

The dataset used (`diabetic_data.csv`) is cleaned and processed to extract relevant features and transform categorical variables into numerical formats, ensuring the data is ready for model training.  

## Key Features:
1. **Data Preprocessing:**  
   - Dropped unnecessary columns such as identifiers and unrelated features.  
   - Mapped categorical data (e.g., gender, medication levels, and test results) to numerical values.  
   - Encoded medication columns using consistent integer mappings to represent medication changes and levels.  
   - Used `LabelEncoder` for encoding other categorical features.  

2. **Feature Scaling and Splitting:**  
   - Divided the dataset into input (`X`) and target (`y`) variables.  
   - Created training, validation, and testing datasets using an 80-20 split, further splitting training data for validation.  

3. **Model Development:**  
   - Built a Multilayer Perceptron (MLP) model using TensorFlow/Keras with:  
     - **Input layer** matching the number of features.  
     - **Two hidden layers** (32 and 16 neurons) with ReLU activation.  
     - **Output layer** with sigmoid activation for binary classification.  
   - Compiled the model with the Adam optimizer, binary cross-entropy loss, and accuracy as the evaluation metric.  

4. **Training and Evaluation:**  
   - Trained the model for 10 epochs using the training and validation datasets.  
   - Evaluated the model's performance on training and testing datasets to calculate accuracy and loss.  

5. **Visualization and Insights:**  
   - Plotted training and validation accuracy/loss to analyze the model’s performance over epochs.  

## Technologies Used:  
- **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib.  
- **Model:** Multilayer Perceptron (MLP) neural network.  
- **Dataset:** `diabetic_data.csv` containing patient records and medical information.  


## Outcome:  
The model successfully predicts the likelihood of patients requiring diabetes medication with good accuracy, showcasing the use of neural networks for binary classification tasks. This project highlights the importance of data preprocessing, feature engineering, and model evaluation in developing robust machine learning solutions.  

