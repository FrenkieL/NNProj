import os
import cv2 as cv
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

IMG_SIZE = 48                 # !!!!! NE MIJENJAJ OVO JER NEĆE CNN RADIT !!!!!!
CATEGORIES = ["REAL", "FAKE"] # ovo je radilo liku: https://github.com/SanKolisetty/AI-Image-Classifier/blob/main/AIImageClassifier.ipynb
DATA_DIR = "./dataset"        # ovo isto

def create_dataset(data_dir=DATA_DIR):
    data = []
    print(f"Loading images from {data_dir}...")
    
    for category in CATEGORIES:
        path = os.path.join(data_dir, category)
        class_num = CATEGORIES.index(category)
        
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist.")
            continue
            
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path, img))
                if img_array is None: continue
                new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([new_array, class_num])
            except Exception as e:
                pass
                
    import random
    random.shuffle(data)
    
    X = []
    y = []
    
    for features, label in data:
        X.append(features)
        y.append(label)
        
    # Normalize and reshape
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    y = np.array(y)
    
    return X, y

def save_pickle(data, filename):
    print(f"Saving {filename}...")
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=4)

def load_pickle(filename):
    print(f"Loading {filename}...")
    with open(filename, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    # lik je slike direktno loadao kako je trenirao model
    # al mi još 2 modela uz ovo moramo pa sam iskoristio pickle
    # malo dodaje bulk u direktorij al nakon ovog prvog pokretanja
    # je bolje nego raw slike ucitavat
    X, y = create_dataset()
    
    # Split here so we have a consistent test set across ALL models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    
    save_pickle(X_train, "X_train.pickle")
    save_pickle(y_train, "y_train.pickle")
    save_pickle(X_test, "X_test.pickle")
    save_pickle(y_test, "y_test.pickle")
    print("Data processing complete.")