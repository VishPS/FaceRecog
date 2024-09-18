import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_images_from_folder(folder_path, image_size=(64, 64)):
    images = []
    filenames = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith(valid_extensions):
                    img_path = os.path.join(subdir_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, image_size)
                        images.append(img.flatten())
                        filenames.append(subdir)  # Use subdir as label
    return np.array(images).T, filenames

def calcmean(Face_Db):
    mean_vector = np.mean(Face_Db, axis=1)
    return mean_vector.reshape(-1, 1)

def savevector(vector, filename):
    np.save(filename, vector)

def normalisation(Face_Db, mean_vector):
    Face_Db_normalized = Face_Db - mean_vector
    return Face_Db_normalized

def covariance(Face_Db_normalized):
    C = np.dot(Face_Db_normalized.T, Face_Db_normalized) / (Face_Db_normalized.shape[1] - 1)
    return C

def eigen(C, k):
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    top_k_eigenvectors = eigenvectors[:, :k]
    return top_k_eigenvectors, eigenvalues

def generate_eigenfaces(Face_Db_normalized, top_k_eigenvectors):
    eigenfaces = np.dot(top_k_eigenvectors.T, Face_Db_normalized)
    return eigenfaces

def generate_face_signatures(Face_Db_normalized, eigenfaces):
    face_signatures = np.dot(eigenfaces.T, Face_Db_normalized)
    return face_signatures

# Update this path to the actual location of the 'faces' folder
dataset_folder = r'M:\assignments\FACERECOG.PY\ann\dataset (1)\dataset\faces'
Face_Db, filenames = load_images_from_folder(dataset_folder)

if Face_Db.size == 0:
    print("No images were loaded.")
else:
    print(f"Face_Db shape: {Face_Db.shape}")
    print(f"Number of images: {Face_Db.shape[1]}")

    mean_vector = calcmean(Face_Db)
    savevector(mean_vector, 'mean_vector.npy')
    
    mean_vector = np.load('mean_vector.npy')
    Face_Db_normalized = normalisation(Face_Db, mean_vector)
    
    C = covariance(Face_Db_normalized)
    
    k = 50
    top_k_eigenvectors, eigenvalues = eigen(C, k)
    
    eigenfaces = generate_eigenfaces(Face_Db_normalized, top_k_eigenvectors)
    
    face_signatures = generate_face_signatures(Face_Db_normalized, eigenfaces)
    
    print("Eigenfaces shape:", eigenfaces.shape)
    print("Face signatures shape:", face_signatures.shape)

# Encode class labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(filenames)
num_classes = len(np.unique(encoded_labels))

# Prepare data for training
train_data, test_data, train_labels, test_labels = train_test_split(
    face_signatures.T, encoded_labels, test_size=0.4, random_state=42
)

# Define and train the ANN
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(k,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer matches num_classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=10, validation_split=0.2)
