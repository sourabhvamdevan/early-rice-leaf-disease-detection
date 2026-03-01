import os
import pickle
import numpy as np
import cv2
import kagglehub
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

IMAGE_SIZE = (224, 224)
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def find_class_folder(base_path):
    for root, dirs, files in os.walk(base_path):
        dirs_with_images = [
            d for d in dirs
            if any(
                f.lower().endswith(('.jpg', '.jpeg', '.png'))
                for f in os.listdir(os.path.join(root, d))
            )
        ]
        if len(dirs_with_images) >= 2:
            print(f"Class folder found at: {root}")
            print(f"Classes: {sorted(dirs_with_images)}")
            return root
    raise FileNotFoundError(
        f"No class folders found inside {base_path}.\n"
        f"Inspect with: import os; [print(r) for r,d,f in os.walk('{base_path}')]"
    )


def load_images(dataset_path):
    class_names = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])
    label_map = {name: idx for idx, name in enumerate(class_names)}
    X, y = [], []

    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        count = 0
        for img_file in os.listdir(class_path):
            if os.path.splitext(img_file)[1].lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            try:
                img = cv2.imread(os.path.join(class_path, img_file))
                img = cv2.resize(img, IMAGE_SIZE)
                img = img / 255.0
                X.append(img)
                y.append(label_map[class_name])
                count += 1
            except Exception as e:
                print(f"  Skipped {img_file}: {e}")
        print(f"  {class_name}: {count} images")

    return np.array(X, dtype=np.float32), np.array(y), class_names, label_map


if __name__ == "__main__":

    # Step 1: Download dataset
    print("Downloading dataset...")
    base_path = kagglehub.dataset_download("nirmalsankalana/rice-leaf-disease-image")
    print(f"Downloaded to: {base_path}")

    # Step 2: Find class folders
    dataset_path = find_class_folder(base_path)

    # Step 3: Load images
    print("\nLoading images...")
    X, y, class_names, label_map = load_images(dataset_path)
    print(f"\nTotal : {len(X)} images")
    print(f"Classes : {class_names}")

    # Step 4: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Step 5: EfficientNetB0 feature extraction
    print("\nLoading EfficientNetB0...")
    base_model = EfficientNetB0(
        weights='imagenet', include_top=False,
        pooling='avg', input_shape=(224, 224, 3)
    )
    extractor = Model(inputs=base_model.input, outputs=base_model.output)
    extractor.trainable = False

    print("Extracting features — train set...")
    X_train_feat = extractor.predict(X_train, batch_size=32, verbose=1)
    print("Extracting features — test set...")
    X_test_feat = extractor.predict(X_test, batch_size=32, verbose=1)
    print(f"Feature shape: {X_train_feat.shape}")

    # Step 6: Train Voting Classifier
    print("\nTraining Voting Classifier (SVM + Random Forest + XGBoost)...")
    clf = VotingClassifier(
        estimators=[
            ('svm', SVC(probability=True, kernel='rbf', C=1.0, random_state=42)),
            ('rf',  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                                  n_estimators=100, random_state=42, n_jobs=-1)),
        ],
        voting='soft'
    )
    clf.fit(X_train_feat, y_train)
    print("Training complete.")

    # Step 7: Evaluate
    print("\nEvaluation on test set:")
    y_pred = clf.predict(X_test_feat)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Step 8: Save everything
    print("Saving model files...")

    with open(f"{MODELS_DIR}/classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    print(f"  Saved: {MODELS_DIR}/classifier.pkl")

    with open(f"{MODELS_DIR}/classes.pkl", "wb") as f:
        pickle.dump(class_names, f)
    print(f"  Saved: {MODELS_DIR}/classes.pkl")

    with open(f"{MODELS_DIR}/label_map.pkl", "wb") as f:
        pickle.dump(label_map, f)
    print(f"  Saved: {MODELS_DIR}/label_map.pkl")

    extractor.save(f"{MODELS_DIR}/efficientnet_extractor.h5")
    print(f"  Saved: {MODELS_DIR}/efficientnet_extractor.h5")

    print(f"\nDone. Download the '{MODELS_DIR}/' folder and place it next to app.py.")
    print(f"Then run:  streamlit run app.py")