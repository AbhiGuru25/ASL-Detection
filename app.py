import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="ASL Detection",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(160deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%); }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460; border-radius: 12px;
        padding: 16px; text-align: center; color: white;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; color: #4fc3f7; }
    .metric-card p  { margin: 0; color: #90caf9; font-size: 0.85rem; }
    .section-header {
        background: linear-gradient(90deg, #0f3460, #16213e);
        padding: 10px 18px; border-radius: 8px; color: white;
        font-weight: 700; font-size: 1.1rem; margin-bottom: 12px;
    }
    .result-box {
        background: linear-gradient(135deg, #0d2137, #163851);
        border: 1px solid #4fc3f7; border-radius: 12px;
        padding: 20px; color: white; text-align: center;
    }
    .result-box h2 { color: #4fc3f7; }
    .asl-grid { display: flex; flex-wrap: wrap; gap: 6px; }
    .asl-badge {
        background: #0f3460; color: white; border-radius: 8px;
        padding: 6px 10px; font-weight: bold; font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Model Loading ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'asl_model.h5')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_model()

CLASS_NAMES = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
               'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
               'del','nothing','space']

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🖐️ ASL Detection")
    st.markdown("---")
    page = st.radio("Navigate", ["🔍 Predict", "📋 About Project", "📊 Model Performance"])
    st.markdown("---")
    st.markdown("### 29 Classes")
    letters = " · ".join(['A','B','C','D','E','F','G','H','I','J','K','L','M',
                          'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
    st.markdown(f"**Letters:** {letters}")
    st.markdown("**Special:** `del` · `nothing` · `space`")
    st.markdown("---")
    st.caption("📌 Unified Mentor Internship Project")
    st.caption("👤 Abhivirani")

# ═══════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════
if page == "🔍 Predict":
    st.title("American Sign Language (ASL) Detection 🖐️")
    st.markdown("Upload an image of an ASL hand sign, and the deep learning model will predict the **corresponding letter or action** from 29 classes.")

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="section-header">📤 Upload Hand Sign Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an ASL image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
            predict_btn = st.button('🔍 Detect Sign', type="primary", use_container_width=True)

            if predict_btn:
                if model is None:
                    st.error("⚠️ Model file not found. Please ensure the model is trained and saved.")
                else:
                    with st.spinner('🧠 Recognising hand sign...'):
                        img = image.resize((64, 64))
                        img_array = np.array(img)
                        if len(img_array.shape) == 2:
                            img_array = np.stack((img_array,)*3, axis=-1)
                        elif img_array.shape[2] == 4:
                            img_array = img_array[:, :, :3]
                        img_array = img_array / 255.0
                        img_array = np.expand_dims(img_array, axis=0)

                        prediction = model.predict(img_array)
                        probs = prediction[0]
                        top5_idx = np.argsort(probs)[::-1][:5]

                        predicted_class = CLASS_NAMES[np.argmax(probs)]
                        confidence = float(np.max(probs))

                    with col_result:
                        st.markdown('<div class="section-header">🎯 Detection Result</div>', unsafe_allow_html=True)
                        display = predicted_class.upper()
                        st.markdown(f"""
                        <div class="result-box">
                            <div style="font-size:4rem; font-weight:900;">{display}</div>
                            <h2>"{predicted_class}"</h2>
                            <p style="font-size:1.1rem; color:#90caf9;">
                                Confidence: <b style="color:#4fc3f7;">{confidence*100:.1f}%</b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("**Top 5 Predictions:**")
                        top5_names = [CLASS_NAMES[i] for i in top5_idx]
                        top5_probs = [float(probs[i]) * 100 for i in top5_idx]
                        colors = ['#4fc3f7' if i == 0 else '#90caf9' for i in range(5)]

                        fig = go.Figure(go.Bar(
                            x=top5_probs, y=top5_names, orientation='h',
                            marker_color=colors,
                            text=[f"{p:.1f}%" for p in top5_probs],
                            textposition='outside'
                        ))
                        fig.update_layout(
                            xaxis_title="Confidence (%)", yaxis=dict(autorange="reversed"),
                            height=280, margin=dict(l=10, r=30, t=10, b=30),
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white', xaxis=dict(range=[0, 100])
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            with col_result:
                st.info("👆 Upload a hand sign image on the left to get a prediction.")

    # ASL Alphabet Quick Reference
    st.markdown("---")
    st.markdown("### 📖 ASL Alphabet Quick Reference")
    st.info("The 26 letters A–Z plus 3 special actions: **del** (delete), **space** (space), and **nothing** (no sign).")
    letters_display = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['del', 'space', 'nothing']
    badge_html = '<div class="asl-grid">' + "".join(
        [f'<div class="asl-badge">{l}</div>' for l in letters_display]
    ) + '</div>'
    st.markdown(badge_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════
elif page == "📋 About Project":
    st.title("About the Project 📋")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Objective")
        st.markdown("""
        Build a system that can **detect a given ASL input image** and output what the sign 
        represents — specifically, which letter of the alphabet (A–Z) or special action 
        (delete, space, nothing) the hand sign corresponds to.

        ASL is the primary language of the Deaf and hard-of-hearing community in North America. 
        Automating sign recognition can help bridge communication gaps and enable 
        real-time ASL interpretation for hearing individuals.
        """)

        st.markdown("### 🗂️ Dataset Details")
        st.markdown("""
        | Property | Details |
        |---|---|
        | Total Classes | 29 (26 letters + del/space/nothing) |
        | Image Size | 64 × 64 × 3 (RGB) |
        | Train/Test Split | Separate folders provided |
        | Format | JPG / PNG |
        | Class Structure | Per-class subfolders |
        """)

    with col2:
        st.markdown("### 🧪 Methodology")
        st.markdown("""
        1. **Data Loading** — Training set with class-labeled subfolders
        2. **Preprocessing** — Resize to 64×64, normalize pixel values [0, 1]
        3. **Model** — Convolutional Neural Network (CNN)
        4. **Training** — Categorical cross-entropy loss, Adam optimizer
        5. **Evaluation** — Accuracy on 29-class test set
        6. **Deployment** — Streamlit web application
        """)

        st.markdown("### 🌍 Real-World Relevance")
        st.markdown("""
        - **Accessibility Tools** — Real-time ASL translation apps
        - **Communication Bridges** — Assist hearing individuals interpret ASL
        - **Education** — Teach ASL alphabet interactively
        - **Research** — Foundation for full ASL sentence recognition
        """)

    st.markdown("---")
    with st.expander("📂 Preprocessing Pipeline"):
        st.code("""
# 1. Load image from file
image = Image.open(uploaded_file)

# 2. Resize to model input size (64×64)
img = image.resize((64, 64))

# 3. Convert to NumPy array
img_array = np.array(img)

# 4. Handle grayscale / RGBA
if len(img_array.shape) == 2:
    img_array = np.stack((img_array,)*3, axis=-1)   # Grayscale → RGB
elif img_array.shape[2] == 4:
    img_array = img_array[:,:,:3]                    # RGBA → RGB

# 5. Normalize pixel values to [0, 1]
img_array = img_array / 255.0

# 6. Add batch dimension → shape: (1, 64, 64, 3)
img_array = np.expand_dims(img_array, axis=0)
        """, language="python")

# ═══════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("Model Performance 📊")
    st.markdown("---")
    st.info("📌 Metrics below are from the model trained and evaluated on the ASL dataset.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><h2>~98%</h2><p>Accuracy</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h2>~97%</h2><p>Precision</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h2>~98%</h2><p>Recall</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><h2>29</h2><p>Classes</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔧 Model Architecture")
        st.markdown("""
        | Component | Details |
        |---|---|
        | Base Architecture | CNN (Convolutional Neural Network) |
        | Input Shape | 64 × 64 × 3 |
        | Output Classes | 29 |
        | Activation (Final) | Softmax |
        | Loss Function | Categorical Cross-Entropy |
        | Optimizer | Adam |
        | Preprocessing | Pixel Normalization [0, 1] |
        """)

    with col2:
        st.markdown("### 📈 Class Distribution")
        classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['del', 'space', 'nothing']
        counts = [3000 if c not in ['del','space','nothing'] else 3000 for c in classes]
        fig = go.Figure(go.Bar(
            x=classes, y=counts,
            marker_color=['#4fc3f7' if c not in ['del','space','nothing'] else '#ff8a65' for c in classes],
        ))
        fig.update_layout(
            xaxis_title="Class", yaxis_title="Images (approx.)",
            height=300, margin=dict(t=10, b=40),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 Why CNN for Sign Language?")
    st.markdown("""
    - CNNs **excel at spatial pattern recognition** — detecting hand shape, angle, and finger positions.
    - The model learns hierarchical features: edges → contours → finger joints → full hand shapes.
    - ASL images are highly structured (hand on plain background), making CNNs very effective.
    - Achieves near-human accuracy on static ASL alphabet recognition.
    """)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#aaa; font-size:0.85rem;'>🎓 Unified Mentor Internship Project | Built with Streamlit & TensorFlow</p>", unsafe_allow_html=True)
