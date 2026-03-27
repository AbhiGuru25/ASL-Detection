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

# ── SVG Icons ─────────────────────────────────────────────────────────────
def icon(svg, size=28, bg="#e8f4fd", color="#1a73e8"):
    return f"""<span style="display:inline-flex;align-items:center;justify-content:center;
        width:{size+12}px;height:{size+12}px;background:{bg};border-radius:10px;
        margin-right:8px;vertical-align:middle;">
        <svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
             viewBox="0 0 24 24" fill="none" stroke="{color}"
             stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{svg}</svg>
    </span>"""

ICO_HAND     = '<path d="M18 11V6a2 2 0 0 0-4 0v-1a2 2 0 0 0-4 0v-1a2 2 0 0 0-4 0v11"/><path d="M6 14.5V17a6 6 0 0 0 12 0v-3"/><line x1="6" y1="14" x2="6" y2="11"/>'
ICO_SEARCH   = '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'
ICO_ABOUT    = '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>'
ICO_CHART    = '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>'
ICO_UPLOAD   = '<polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>'
ICO_TARGET   = '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'
ICO_CPU      = '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'
ICO_LAYERS   = '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>'
ICO_GRID     = '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>'

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(160deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%); }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .metric-card { background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460; border-radius: 12px;
        padding: 16px; text-align: center; color: white; }
    .metric-card h2 { font-size: 2rem; margin: 0; color: #4fc3f7; }
    .metric-card p  { margin: 0; color: #90caf9; font-size: 0.85rem; }
    .section-header { display:flex; align-items:center;
        background: linear-gradient(90deg, #0f3460, #16213e);
        padding: 10px 18px; border-radius: 8px; color: white;
        font-weight: 700; font-size: 1.1rem; margin-bottom: 12px; }
    .result-box { background: linear-gradient(135deg, #0d2137, #163851);
        border: 1px solid #4fc3f7; border-radius: 12px;
        padding: 20px; color: white; text-align: center; }
    .result-box h2 { color: #4fc3f7; }
    .asl-badge { display:inline-block; background:#0f3460; color:white;
        border-radius:8px; padding:6px 12px; font-weight:bold;
        font-size:0.95rem; margin:3px; }
    .page-title { display:flex; align-items:center; gap:10px; margin-bottom:0.5rem; }
    .page-title h1 { margin:0; }
</style>
""", unsafe_allow_html=True)

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

with st.sidebar:
    st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;padding:4px 0 12px 0;">
        {icon(ICO_HAND, 26, '#1a3a5c', '#4fc3f7')}
        <span style="font-size:1.1rem;font-weight:700;color:white;">ASL Detection</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", ["Predict", "About Project", "Model Performance"])
    st.markdown("---")
    st.markdown("<p style='color:#90caf9;font-size:0.8rem;font-weight:600;'>29 CLASSES</p>", unsafe_allow_html=True)
    letters = " · ".join(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    st.markdown(f"<span style='color:#e0e0e0;font-size:0.85rem;'><b>Letters:</b> {letters}</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:#e0e0e0;font-size:0.85rem;'><b>Special:</b> del · nothing · space</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Unified Mentor Internship Project")
    st.caption("Abhivirani")

# ══════════════════════════════════════════════════════════
if page == "Predict":
    st.markdown(f"""<div class="page-title">
        {icon(ICO_HAND, 30, '#e8f4fd', '#1a73e8')}
        <h1>American Sign Language (ASL) Detection</h1></div>""", unsafe_allow_html=True)
    st.markdown("Upload an image of an ASL hand sign — the model will predict the **corresponding letter or action** from 29 classes.")

    col_upload, col_result = st.columns([1, 1], gap="large")
    with col_upload:
        st.markdown(f'<div class="section-header">{icon(ICO_UPLOAD, 18, "transparent", "#90caf9")} Upload Hand Sign Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an ASL image", type=["jpg","jpeg","png"], label_visibility="collapsed")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
            predict_btn = st.button('Detect Sign', type="primary", use_container_width=True)

            if predict_btn:
                if model is None:
                    st.error("Model file not found. Please ensure the model is trained and saved.")
                else:
                    with st.spinner('Recognising hand sign...'):
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
                        st.markdown(f'<div class="section-header">{icon(ICO_TARGET, 18, "transparent", "#90caf9")} Detection Result</div>', unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="result-box">
                            <div style="font-size:3.5rem;font-weight:900;color:#4fc3f7;">{predicted_class.upper()}</div>
                            <h2>"{predicted_class}"</h2>
                            <p style="font-size:1.1rem;color:#90caf9;">Confidence: <b style="color:#4fc3f7;">{confidence*100:.1f}%</b></p>
                        </div>""", unsafe_allow_html=True)
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
                st.info("Upload a hand sign image on the left to get a prediction.")

    st.markdown("---")
    st.markdown(f"### {icon(ICO_GRID, 20, '#e8f4fd', '#1a73e8')} ASL Alphabet Quick Reference", unsafe_allow_html=True)
    st.info("26 letters A–Z plus 3 special actions: **del** (delete), **space**, and **nothing** (no sign).")
    letters_display = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['del', 'space', 'nothing']
    badge_html = "".join([f'<span class="asl-badge">{l}</span>' for l in letters_display])
    st.markdown(badge_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
elif page == "About Project":
    st.markdown(f"""<div class="page-title">
        {icon(ICO_ABOUT, 30, '#e8f4fd', '#1a73e8')}
        <h1>About the Project</h1></div>""", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {icon(ICO_TARGET, 20, '#e8f4fd', '#1a73e8')} Objective", unsafe_allow_html=True)
        st.markdown("""
        Build a system that can **detect a given ASL input image** and output what the sign represents —
        which letter (A–Z) or special action (delete, space, nothing) the hand sign corresponds to.

        ASL is the primary language of the Deaf and hard-of-hearing community in North America.
        Automating sign recognition helps bridge communication gaps.
        """)
        st.markdown(f"### {icon(ICO_LAYERS, 20, '#e8f4fd', '#1a73e8')} Dataset Details", unsafe_allow_html=True)
        st.markdown("""
        | Property | Details |
        |---|---|
        | Total Classes | 29 (A–Z + del/space/nothing) |
        | Image Size | 64 × 64 × 3 (RGB) |
        | Train/Test Split | Separate folders |
        | Format | JPG / PNG |
        """)

    with col2:
        st.markdown(f"### {icon(ICO_CPU, 20, '#e8f4fd', '#1a73e8')} Methodology", unsafe_allow_html=True)
        st.markdown("""
        1. **Data Loading** — Training set with class-labeled subfolders
        2. **Preprocessing** — Resize to 64×64, normalize pixel values [0, 1]
        3. **Model** — Convolutional Neural Network (CNN)
        4. **Training** — Categorical cross-entropy loss, Adam optimizer
        5. **Evaluation** — Accuracy on 29-class test set
        6. **Deployment** — Streamlit web application
        """)
        st.markdown(f"### {icon(ICO_CHART, 20, '#e8f4fd', '#1a73e8')} Real-World Relevance", unsafe_allow_html=True)
        st.markdown("""
        - **Accessibility Tools** — Real-time ASL translation apps
        - **Communication Bridges** — Assist hearing individuals interpret ASL
        - **Education** — Teach ASL alphabet interactively
        - **Research** — Foundation for full ASL sentence recognition
        """)

    with st.expander("Preprocessing Pipeline"):
        st.code("""
image = Image.open(uploaded_file)
img = image.resize((64, 64))
img_array = np.array(img)
if len(img_array.shape) == 2:
    img_array = np.stack((img_array,)*3, axis=-1)
elif img_array.shape[2] == 4:
    img_array = img_array[:,:,:3]
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 64, 64, 3)
        """, language="python")

# ══════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown(f"""<div class="page-title">
        {icon(ICO_CHART, 30, '#e8f4fd', '#1a73e8')}
        <h1>Model Performance</h1></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.info("Metrics from the model trained and evaluated on the ASL dataset.")

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
        st.markdown(f"### {icon(ICO_CPU, 20, '#e8f4fd', '#1a73e8')} Model Architecture", unsafe_allow_html=True)
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
        st.markdown(f"### {icon(ICO_LAYERS, 20, '#e8f4fd', '#1a73e8')} Why CNN for Sign Language?", unsafe_allow_html=True)
        st.markdown("""
        - CNNs excel at **spatial pattern recognition** — hand shapes, angles, finger positions.
        - Learns hierarchical features: edges → contours → finger joints → full hand shapes.
        - Achieves near-human accuracy on static ASL alphabet recognition.
        """)
    with col2:
        st.markdown(f"### {icon(ICO_CHART, 20, '#e8f4fd', '#1a73e8')} Class Distribution", unsafe_allow_html=True)
        classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['del','space','nothing']
        colors_bar = ['#4fc3f7' if c not in ['del','space','nothing'] else '#ff8a65' for c in classes]
        fig = go.Figure(go.Bar(x=classes, y=[3000]*29, marker_color=colors_bar))
        fig.update_layout(
            xaxis_title="Class", yaxis_title="Images (approx.)",
            height=300, margin=dict(t=10, b=40),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#aaa;font-size:0.85rem;'>Unified Mentor Internship Project | Built with Streamlit & TensorFlow</p>", unsafe_allow_html=True)
