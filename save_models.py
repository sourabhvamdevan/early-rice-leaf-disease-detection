"""
save_models.py

Run this script in your Colab/Jupyter environment AFTER training
to save your models so Streamlit can load them.

Usage (in Colab, after training):
    !python save_models.py
"""

import pickle
import os

os.makedirs("models", exist_ok=True)


try:
    with open("models/classifier.pkl", "wb") as f:
        pickle.dump(voting_clf, f)

    with open("models/classes.pkl", "wb") as f:
        pickle.dump(list(labels), f)

    print(" Saved: models/classifier.pkl")
    print(" Saved: models/classes.pkl")
    print("Class names saved:", list(labels))
except NameError as e:
    print(f"  Variable not found: {e}")
    print("Make sure you run this after training `voting_clf` and defining `labels`.")


try:
    feature_extractor.save("models/efficientnet_extractor.h5")
    print(" Saved: models/efficientnet_extractor.h5")
except NameError:
    print(" `feature_extractor` not found — skipping. The app will auto-download EfficientNetB0 weights.")
