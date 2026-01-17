import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import data_processing
from CNN_body import AIImageDetector
from SVMs import SVMManager

def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[{name}] Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred))
    return acc

def plot_comparison(results):
    names = list(results.keys())
    scores = list(results.values())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=names, y=scores, palette="viridis")
    plt.ylim(0, 1.0)
    plt.title("Model Accuracy Comparison (CIFAKE)")
    plt.ylabel("Accuracy")
    plt.savefig("model_comparison.png")
    print("Graph saved to model_comparison.png")
    plt.show()

if __name__ == "__main__":
    print("--- Loading Test Data ---")
    # We need Raw images for CNN, Features for SVM
    X_test_raw = data_processing.load_pickle("X_test.pickle")
    X_test_feats = data_processing.load_pickle("X_test_features.pickle")
    y_test = data_processing.load_pickle("y_test.pickle")
    
    results = {}

    # 1. Evaluate CNN
    print("\n--- Evaluating CNN ---")
    cnn = AIImageDetector()
    cnn.load_weights("cnn_weights.h5")
    # CNN outputs probabilities, threshold at 0.5
    cnn_preds_prob = cnn.model.predict(X_test_raw)
    cnn_preds = (cnn_preds_prob > 0.5).astype("int32").flatten()
    results["CNN"] = evaluate_model("CNN", y_test, cnn_preds)

    # 2. Evaluate SVMs
    svm_types = ["linear", "rbf"]
    
    for k in svm_types:
        try:
            print(f"\n--- Evaluating SVM ({k}) ---")
            svm = SVMManager(k)
            svm.load(f"svm_{k}.joblib")
            preds = svm.predict(X_test_feats)
            results[f"SVM_{k}"] = evaluate_model(f"SVM-{k}", y_test, preds)
        except Exception as e:
            print(f"Could not load SVM_{k}: {e}")

    # 3. Visualize
    plot_comparison(results)