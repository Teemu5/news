from utils import get_models
from recommender import ensemble_bagging, ensemble_boosting, ensemble_stacking, hybrid_ensemble
import numpy as np

def main():
    models = get_models()
    test_input = "This is a dummy test input for the recommender function."
    bagging_pred = ensemble_bagging(test_input, models)
    print("Bagging Prediction:", bagging_pred)
    dummy_errors = np.array([0.2, 0.15, 0.25])
    boosting_pred = ensemble_boosting(test_input, models, dummy_errors)
    print("Boosting Prediction:", boosting_pred)
    X_train_dummy = np.array([
        [0.80, 0.75, 0.85],
        [0.55, 0.60, 0.50],
        [0.30, 0.35, 0.25],
        [0.20, 0.25, 0.15]
    ])
    y_train_dummy = np.array([1, 0, 1, 0])
    from recommender import train_stacking_meta_model
    meta_model = train_stacking_meta_model(X_train_dummy, y_train_dummy)
    stacking_pred = ensemble_stacking(test_input, models, meta_model)
    print("Stacking Prediction:", stacking_pred)
    hybrid_pred = hybrid_ensemble(test_input, models, dummy_errors, meta_model)
    print("Hybrid Ensemble Prediction:", hybrid_pred)

if __name__ == "__main__":
    main()
