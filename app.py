import streamlit as st
import numpy as np
import cv2
import os
import pywt
import matplotlib.pyplot as plt
from PIL import Image
import pickle

st.set_page_config(
    page_title="Rice Leaf Disease Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title { font-size:2.4rem; font-weight:700; color:#2e7d32; text-align:center; margin-bottom:0.2rem; }
    .subtitle   { text-align:center; color:#666; font-size:1.05rem; margin-bottom:1.5rem; }
    .result-card { background:linear-gradient(135deg,#e8f5e9,#f1f8e9); border-left:6px solid #43a047;
                   border-radius:10px; padding:1.4rem; margin-top:1rem; }
    .warning-card { background:#fff3e0; border-left:6px solid #fb8c00;
                    border-radius:10px; padding:1.2rem; margin-top:1rem; }
    .healthy-card { background:#e3f2fd; border-left:6px solid #1e88e5;
                    border-radius:10px; padding:1.2rem; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)

IMAGE_SIZE = (224, 224)
MODELS_DIR = "models"

DISEASE_INFO = {
    "Bacterial leaf blight": {
        "description": "Caused by Xanthomonas oryzae. Produces water-soaked lesions that turn yellow then white along leaf margins.",
        "treatment":   "Use resistant varieties. Apply copper-based bactericides. Drain fields and avoid excessive nitrogen.",
        "severity":    "High",
    },
    "Brown spot": {
        "description": "Caused by Cochliobolus miyabeanus. Produces brown oval lesions with yellow halos on leaves and grains.",
        "treatment":   "Treat seeds with fungicides. Apply mancozeb or iprodione. Maintain balanced soil nutrition.",
        "severity":    "Moderate",
    },
    "Leaf smut": {
        "description": "Caused by Entyloma oryzae. Forms small angular black spots on both surfaces of the leaf blade.",
        "treatment":   "Use certified disease-free seeds. Apply propiconazole fungicide. Remove and destroy infected debris.",
        "severity":    "Low",
    },
}

def get_info(class_name):
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
    key = next((k for k in DISEASE_INFO if k.lower() == class_name.lower()), None)
    if key:
        return DISEASE_INFO[key]
    return {
        "description": "Class: " + class_name,
        "treatment":   "Consult a local agronomist for treatment recommendations.",
        "severity":    "Unknown",
    }

@st.cache_resource(show_spinner="Loading model...")
def load_models():
    clf_path     = os.path.join(MODELS_DIR, "classifier.pkl")
    classes_path = os.path.join(MODELS_DIR, "classes.pkl")
    h5_path      = os.path.join(MODELS_DIR, "efficientnet_extractor.h5")

    if not (os.path.exists(clf_path) and os.path.exists(classes_path)):
        return None, None, None

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    with open(classes_path, "rb") as f:
        class_names = pickle.load(f)

    try:
        import tensorflow as tf
        if os.path.exists(h5_path):
            extractor = tf.keras.models.load_model(h5_path, compile=False)
        else:
            from tensorflow.keras.applications import EfficientNetB0
            from tensorflow.keras.models import Model
            base = EfficientNetB0(weights='imagenet', include_top=False,
                                  pooling='avg', input_shape=(224, 224, 3))
            extractor = Model(inputs=base.input, outputs=base.output)
    except Exception as e:
        st.error("Could not load EfficientNet: " + str(e))
        extractor = None

    return clf, class_names, extractor

clf, class_names, extractor = load_models()

def predict(image):
    img      = cv2.resize(np.array(image), IMAGE_SIZE).astype(np.float32) / 255.0
    features = extractor.predict(np.expand_dims(img, axis=0), verbose=0)
    probs    = clf.predict_proba(features)[0] * 100
    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx], probs, class_names

with st.sidebar:
    st.markdown("## Rice Leaf Disease Detector")
    st.markdown("---")
    if clf is not None:
        st.success("Model loaded successfully")
        st.markdown("**Classes (" + str(len(class_names)) + "):**")
        for c in class_names:
            info = get_info(c)
            st.markdown("- **" + c + "** — " + info["severity"])
    else:
        st.error("No model found in models/")
        st.markdown("Run the notebook cells in Colab, download models.zip, extract it and place the models/ folder next to app.py.")
    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown(
        "- **Feature extraction:** EfficientNetB0\n"
        "- **Classifier:** VotingClassifier\n"
        "  - SVM (RBF kernel)\n"
        "  - Random Forest (100 trees)\n"
        "  - XGBoost"
    )

st.markdown('<div class="main-title">Rice Leaf Disease Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a rice leaf image for instant AI-powered disease diagnosis</div>', unsafe_allow_html=True)

if clf is None:
    st.error("Model files not found in models/.")
    st.markdown("""
**To fix this:**
1. Open `final_codee_updated.ipynb` in Google Colab
2. Run both cells
3. Extract the downloaded `models.zip`
4. Place the `models/` folder next to `app.py`
5. Restart the app
    """)
    st.stop()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    uploaded = st.file_uploader("Upload a rice leaf image", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.success("Uploaded: " + uploaded.name)
        analyze_btn = st.button("Analyze Leaf", type="primary", use_container_width=True)

with col2:
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

if uploaded and analyze_btn:
    with st.spinner("Analyzing leaf..."):
        pred_cls, probs, classes = predict(image)

    confidence = float(np.max(probs))
    info = get_info(pred_cls)

    st.markdown("---")
    st.markdown("## Prediction Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Disease", pred_cls)
    m2.metric("Confidence",        str(round(confidence, 1)) + "%")
    m3.metric("Severity",          info["severity"])
    m4.metric("Classes Evaluated", str(len(classes)))

    if confidence < 60:
        st.markdown(
            '<div class="warning-card"><strong>Early Detection Alert:</strong> '
            'Confidence is below 60%. Symptoms may be in early stages. '
            'Monitor closely and consult an agronomist before applying treatment.</div>',
            unsafe_allow_html=True,
        )

    card_class = "healthy-card" if "healthy" in pred_cls.lower() else "result-card"
    st.markdown(
        '<div class="' + card_class + '">'
        '<h4>' + pred_cls + '</h4>'
        '<p><strong>Description:</strong> ' + info["description"] + '</p>'
        '<p><strong>Recommended Treatment:</strong> ' + info["treatment"] + '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Confidence Scores")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor('#fafafa')

    colors = ["#43a047" if c == pred_cls else "#90caf9" for c in classes]
    bars   = axes[0].barh(classes, probs, color=colors, edgecolor='white')
    axes[0].set_xlabel("Confidence (%)", fontsize=10)
    axes[0].set_title("Per-Class Confidence", fontsize=11, fontweight='bold')
    axes[0].set_xlim(0, 115)
    axes[0].set_facecolor('#fafafa')
    for bar, val in zip(bars, probs):
        axes[0].text(val + 1, bar.get_y() + bar.get_height() / 2,
                     str(round(val, 1)) + "%", va='center', fontsize=9)

    wedge_colors = ["#43a047" if c == pred_cls else plt.cm.Pastel1(i / len(classes))
                    for i, c in enumerate(classes)]
    axes[1].pie(probs, labels=classes, autopct='%1.1f%%', startangle=90,
                colors=wedge_colors, pctdistance=0.8, textprops={'fontsize': 9})
    axes[1].set_title("Probability Distribution", fontsize=11, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("Wavelet Feature Decomposition"):
        img_array  = np.array(image)
        gray       = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray_small = cv2.resize(gray, (128, 128))
        coeffs     = pywt.wavedec2(gray_small, 'db1', level=2, mode='periodization')
        titles     = ["LL (Approx)", "LH (Horizontal)", "HL (Vertical)", "HH (Diagonal)"]
        imgs_wv    = [coeffs[0], coeffs[1][0], coeffs[1][1], coeffs[1][2]]
        fig2, axes2 = plt.subplots(1, 4, figsize=(14, 3))
        for ax, wv_img, title in zip(axes2, imgs_wv, titles):
            ax.imshow(np.abs(wv_img), cmap='viridis', aspect='auto')
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        plt.suptitle("2D Wavelet Decomposition - Level 1 subbands", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

elif not uploaded:
    st.markdown("""
    <div style='text-align:center; padding:3rem; background:#f9fbe7;
         border-radius:12px; border:2px dashed #aed581; margin-top:1rem;'>
        <h3 style='color:#558b2f;'>Upload a rice leaf image to begin</h3>
        <p style='color:#777;'>Supports JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)
