from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib

def train_models(X_train, X_test, y_train, y_test):

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)

    # Random Forest with Hyperparameter Tuning
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    }

    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
    grid.fit(X_train, y_train)

    rf = grid.best_estimator_
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    print(f"Logistic Regression Accuracy: {lr_acc*100:.2f}%")
    print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
    print("Best Parameters:", grid.best_params_)

    # Confusion Matrix
    cm = confusion_matrix(y_test, rf_pred)
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    # ROC Curve
    y_prob = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("models/roc_curve.png")
    plt.close()

    # Feature Importance
    importances = rf.feature_importances_
    plt.bar(range(len(importances)), importances)
    plt.title("Feature Importance")
    plt.savefig("models/feature_importance.png")
    plt.close()

    joblib.dump(rf, "models/best_model.pkl")

    return lr_acc, rf_acc
