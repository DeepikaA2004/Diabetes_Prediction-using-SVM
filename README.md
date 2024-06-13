# Diabetes Prediction

## Introduction

The "Diabetes Prediction Using Support Vector Machine" project aims to harness the power of machine learning to predict the likelihood of diabetes in individuals based on various health indicators. Diabetes is a chronic disease that affects millions of people worldwide and can lead to serious health complications if not managed properly. Early detection and preventive measures are crucial for mitigating the risks associated with diabetes.

Support Vector Machine (SVM) is a supervised machine learning algorithm known for its effectiveness in classification tasks. In this project, we employ SVM to analyze a dataset containing various features related to health and lifestyle, such as glucose levels, blood pressure, body mass index (BMI), and other relevant factors. By training the SVM model on this dataset, we aim to develop a reliable predictive tool that can assist healthcare professionals in identifying individuals at risk of developing diabetes.

## Data Collection and Analysis

### PIMA Diabetes Dataset
The code begins by loading the PIMA Diabetes Dataset into a pandas DataFrame. It prints the first 5 rows of the dataset, displays the number of rows and columns, and provides statistical measures of the data. The target variable, 'Outcome,' is analyzed, where 0 indicates non-diabetic and 1 indicates diabetic.

## Data Standardization
The features are standardized using StandardScaler from scikit-learn.This ensures that all features have a mean of 0 and a standard deviation of 1.

## Train Test Split
The dataset is split into training and testing sets using the train_test_split function. The split is performed with a test size of 20%, maintaining the stratification of the outcome variable.

## Training the Model
The code utilizes the SVM algorithm for classification. It creates an SVM classifier with a linear kernel and trains it on the training data.

## Model Evaluation
The accuracy scores are calculated for both the training and testing datasets. The accuracy on the training data is approximately 78.66%, and on the testing data, it is around 77.27%.

## Making a Predictive System
A predictive system is created to make predictions on new input data. The input data is standardized using the same scaler, and the trained SVM classifier predicts whether the person is diabetic or not.

### Example Prediction
An example prediction is provided using input data (5, 166, 72, 19, 175, 25.8, 0.587, 51). The standardized input data is predicted, and based on the prediction, it is determined whether the person is diabetic or not.

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Scikit-learn

## Model Deployment screenshots
![Screenshot (38)](https://github.com/DeepikaA2004/Diabetes_Prediction/assets/110418508/ced7632a-3c4c-4752-8535-225353fba6fb)

![Screenshot (39)](https://github.com/DeepikaA2004/Diabetes_Prediction/assets/110418508/af482b53-8cc2-42bb-946d-9ff1b9d43428)

## Conclusion

The "Diabetes Prediction Using Support Vector Machine" project successfully demonstrates the application of machine learning techniques to predict the likelihood of diabetes in individuals. By leveraging the Support Vector Machine (SVM) algorithm, the project achieved a high level of accuracy in classifying individuals based on their risk of developing diabetes, thus providing a valuable tool for early detection and preventive healthcare.

The performance of the SVM model, as indicated by the provided output images and performance graphs, highlights its effectiveness in handling this classification task. The use of SVM, known for its robustness and efficiency, proves to be a reliable method for diabetes prediction, offering significant benefits in medical diagnostics.

Future work could involve further optimizing the model by incorporating a larger and more diverse dataset, exploring different kernel functions for SVM to enhance accuracy, and integrating additional features that may contribute to diabetes risk assessment. Additionally, deploying the model in a real-world healthcare setting, such as a mobile application or a web-based tool, could make it accessible to a broader audience, aiding in early diagnosis and personalized health management.

Overall, this project underscores the potential of machine learning in transforming healthcare by providing accurate and timely predictions. The implementation of SVM for diabetes prediction not only demonstrates technological innovation but also contributes to the broader goal of improving public health outcomes through data-driven insights.

## Contact

**MY LINKEDIN PROFILE** - https://www.linkedin.com/in/deepika2004/
