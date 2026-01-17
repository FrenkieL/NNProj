import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import data_processing

class CNN:
    def __init__(self, input_shape=(48, 48, 3)):
        self.input_shape = input_shape
        self.model = self._build_model()
        self.feature_extractor = None

    def _build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # vektorizacija uz minimalno izmjena <3
            layers.Flatten(name="flatten_layer"),
            
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y, epochs=15, batch_size=64):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    def save_weights(self, path="cnn_weights.h5"):
        self.model.save(path)

    def load_weights(self, path="cnn_weights.h5"):
        self.model = models.load_model(path)
        self._init_extractor()

    def _init_extractor(self):
        self.feature_extractor = models.Model(
            inputs=self.model.input, 
            outputs=self.model.get_layer("flatten_layer").output
        )

    def extract_features(self, X):
        if self.feature_extractor is None:
            self._init_extractor()
        # batches da nekom ne izgori komp (meni)
        return self.feature_extractor.predict(X, batch_size=64)

if __name__ == "__main__":
    X_train = data_processing.load_pickle("X_train.pickle")
    y_train = data_processing.load_pickle("y_train.pickle")
    X_test = data_processing.load_pickle("X_test.pickle")
    
    model = CNN()
    model.train(X_train, y_train, epochs=15)
    model.save_weights("cnn_weights.h5")
    
    print("Extracting features for SVMs...")
    X_train_feats = model.extract_features(X_train)
    X_test_feats = model.extract_features(X_test)
    
    data_processing.save_pickle(X_train_feats, "X_train_features.pickle")
    data_processing.save_pickle(X_test_feats, "X_test_features.pickle")
    print(f"Features saved. Shape: {X_train_feats.shape}")