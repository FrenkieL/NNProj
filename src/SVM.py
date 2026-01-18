import numpy as np
import data_preprocessing as data_processing
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import joblib

class SVMManager:
    def __init__(self, kernel_type='linear'):
        self.kernel_type = kernel_type
        self.model = None
        
    def train(self, X_features, y, optimize=False):
        print(f"\n--- Training {self.kernel_type.upper()} SVM ---")
        
        if self.kernel_type == 'linear':
            base_model = LinearSVC(dual=False, max_iter=2000)
            params = {'C': [0.1, 1, 10]}
            
        elif self.kernel_type == 'rbf':
            base_model = SVC(kernel='rbf')
            params = {'C': [1, 10], 'gamma': ['scale', 0.1]}

        if optimize:
            print("grid search for rbf...")
            # tko god da ce testirati i trenirati nek smanji ove max features ja sam samo bubnuo broj
            if len(X_features) > 5000:
                idx = np.random.choice(len(X_features), 5000, replace=False)
                X_sub, y_sub = X_features[idx], y[idx]
            else:
                X_sub, y_sub = X_features, y
                
            grid = GridSearchCV(base_model, params, cv=3, n_jobs=-1, verbose=1)
            grid.fit(X_sub, y_sub)
            print(f"Best Params: {grid.best_params_}")
            self.model = grid.best_estimator_
        else:
            self.model = base_model
            
        print("Fitting final model on full dataset...")
        self.model.fit(X_features, y)
        
    def predict(self, X_features):
        return self.model.predict(X_features)

    def save(self, filename):
        joblib.dump(self.model, filename)

    def load(self, filename):
        self.model = joblib.load(filename)

if __name__ == "__main__":
    # za ovo se mora napravit preprocessing -> vidi main u CNN_body.py 
    # inace bi trebali vektorizirat slike al cemu kad vec CNN baci vektor prije softmaxa
    X_train_feats = data_preprocessing.load_pickle("X_train_features.pickle")
    y_train = data_preprocessing.load_pickle("y_train.pickle")
    
    # linearni
    svm_lin = SVMManager('linear')
    svm_lin.train(X_train_feats, y_train, optimize=True)
    svm_lin.save("svm_linear.joblib")
    
    # rbf
    svm_rbf = SVMManager('rbf')
    svm_rbf.train(X_train_feats, y_train, optimize=True)
    svm_rbf.save("svm_rbf.joblib")