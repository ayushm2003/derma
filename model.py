import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Function to train and test a Naive Bayes classifier
def train_naive_bayes(file_path):
	# Load the dataset
	data = pd.read_csv(file_path)

	# Encoding categorical variables to numeric values
	label_encoders = {}
	for column in data.columns:
		encoder = LabelEncoder()
		data[column] = encoder.fit_transform(data[column])
		label_encoders[column] = encoder

	# Features and target variable
	X = data.iloc[:, 1:]  # symptoms columns
	y = data.iloc[:, 0]   # disease column

	# Splitting the dataset into the Training set and Test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	# Training the Naive Bayes model on the Training set
	classifier = MultinomialNB()
	classifier.fit(X_train, y_train)

	# Predicting the Test set results and evaluate
	y_pred = classifier.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)

	return classifier, label_encoders, accuracy, data.columns

# Function to predict the disease given a list of symptoms
def predict_disease(symptoms, label_encoders, classifier, data_columns):
	# Initialize an array for encoded symptoms with '0's for absent symptoms
	encoded_symptoms = np.zeros(len(data_columns)-1, dtype=int)

	# Update the array with encoded values for present symptoms
	for i, symptom in enumerate(symptoms):
		if symptom in label_encoders[data_columns[i+1]].classes_:
			encoded_symptoms[i] = label_encoders[data_columns[i+1]].transform([symptom])[0]

	# Reshape input data to match the number of features
	encoded_symptoms = encoded_symptoms.reshape(1, -1)

	# Predicting the disease
	predicted_disease_encoded = classifier.predict(encoded_symptoms)

	# Decoding the prediction to the original disease label
	predicted_disease = label_encoders['Disease'].inverse_transform(predicted_disease_encoded)

	return predicted_disease[0]

# File path to the dataset
file_path = 'datasets/DiseaseAndSymptoms.csv'

# Train the classifier
classifier, label_encoders, accuracy, data_columns = train_naive_bayes(file_path)

# Print the accuracy of the model
print(f"Model accuracy: {accuracy:.2%}")

# Example of using the function to predict a disease
example_symptoms = ['fever', 'headache', 'vomiting', 'nausea']
predicted_disease = predict_disease(example_symptoms, label_encoders, classifier, data_columns)
print(predicted_disease)
