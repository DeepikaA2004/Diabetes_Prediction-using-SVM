# Diabetes Prediction

 This project uses the PIMA Diabetes Dataset. The model is built using the Support Vector Machine (SVM) algorithm. Below is a comprehensive guide on the dataset, data analysis, model training, evaluation, and making predictions.

## Data Collection and Analysis

### PIMA Diabetes Dataset
The code begins by loading the PIMA Diabetes Dataset into a pandas DataFrame. It prints the first 5 rows of the dataset, displays the number of rows and columns, and provides statistical measures of the data. The target variable, 'Outcome,' is analyzed, where 0 indicates non-diabetic and 1 indicates diabetic.

## Data Standardization
The features are standardized using StandardScaler from scikit-learn. This ensures that all features have a mean of 0 and a standard deviation of 1.

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

## License
This project is licensed under the [MIT License](LICENSE).
