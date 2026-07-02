from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def _fit_and_score_models(X_train, X_test, y_train, y_test, random_state=42):
    results = {}

    rfc = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    results["RandomForest"] = classification_report(y_test, y_pred, output_dict=True)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        random_state=random_state,
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    results["MLP"] = classification_report(y_test, y_pred, output_dict=True)

    svm = LinearSVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    results["SVM"] = classification_report(y_test, y_pred, output_dict=True)

    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    results["XGBoost"] = classification_report(y_test, y_pred, output_dict=True)

    return results


def train_and_evaluate_lisa_models(features, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    return _fit_and_score_models(
        X_train, X_test, y_train, y_test, random_state=random_state
    )


def train_and_evaluate_lisa_models_ratios(
    features,
    labels,
    ratios=None,
    test_size=0.2,
    random_state=42,
):
    if ratios is None:
        ratios = [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]

    results_ratios = {}
    labels_array = np.array(labels)
    ai_indices = np.where(labels_array == 1)[0]
    human_indices = np.where(labels_array == 0)[0]
    rng = np.random.default_rng(seed=random_state)

    for ai_ratio, human_ratio in ratios:
        print(f"\nEvaluating LISA models with AI:Human ratio = {ai_ratio}:{human_ratio}")

        ai_sample_size = int(
            min(len(ai_indices) / ai_ratio, len(human_indices) / human_ratio) * ai_ratio
        )
        human_sample_size = int(
            min(len(ai_indices) / ai_ratio, len(human_indices) / human_ratio) * human_ratio
        )

        ai_sample_indices = rng.choice(ai_indices, size=ai_sample_size, replace=False)
        human_sample_indices = rng.choice(
            human_indices, size=human_sample_size, replace=False
        )

        selected_indices = np.concatenate([ai_sample_indices, human_sample_indices])
        rng.shuffle(selected_indices)

        custom_features = features[selected_indices]
        custom_labels = labels_array[selected_indices]

        X_train, X_test, y_train, y_test = train_test_split(
            custom_features,
            custom_labels,
            test_size=test_size,
            stratify=custom_labels,
            random_state=random_state,
        )

        results = _fit_and_score_models(
            X_train, X_test, y_train, y_test, random_state=random_state
        )
        results_ratios[f"AI_{int(ai_ratio * 100)}_Human_{int(human_ratio * 100)}"] = results

        final_counts = Counter(custom_labels)
        print("Final class distribution:")
        for label, count in final_counts.items():
            label_name = "AI" if label == 1 else "Human"
            print(f"{label_name}: {count}")

    return results_ratios
