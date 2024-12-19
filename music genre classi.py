# Import necessary libraries
import librosa
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Function to extract features from an audio file
def extract_features(file_name):
    try:
        # Load the audio file using Librosa
        audio, sample_rate = librosa.load(file_name, sr=None, res_type='kaiser_fast') 
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)  # Take the mean of each MFCC feature over time
        
        return mfccs_mean
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

# Initialize lists to hold dataset and labels
dataset = []
labels = []

# Define the path to your dataset and the genres you want to classify
path_to_data = r"C:\Users\adith\OneDrive\Desktop\GTZAN dataset\Data\genres_original" # Replace with your data path
genres = ['classical', 'jazz', 'rock']  # Replace with your genres

# Iterate over each genre and extract features from the audio files
for genre in genres:
    genre_path = os.path.join(path_to_data, genre)
    for file in os.listdir(genre_path):
        if file.endswith('.wav'):  # Adjust if your files have a different extension
            file_path = os.path.join(genre_path, file)
            data = extract_features(file_path)
            if data is not None:  # Only add if features were successfully extracted
                dataset.append(data)
                labels.append(genre)

# Create a DataFrame from the extracted features and labels
df = pd.DataFrame(dataset)
df['label'] = labels

# Split the data into training and testing sets
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the K-Nearest Neighbors (KNN) classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Compute and display the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()

# Show the plot
plt.show()
