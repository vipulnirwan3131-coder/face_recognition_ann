"""
app.py
-------
Streamlit web interface for the ANN-based Face Recognition System.
Premium dashboard UI with interactive Plotly visualizations + real-time
face detection on the webcam feed.

Run:
    streamlit run app.py
    (or, if PATH issues occur:  python -m streamlit run app.py)
"""

import os
import time
import hashlib
import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from utils.face_utils import (
    ensure_dirs, list_registered_people, count_images, total_images,
    load_last_metrics, MODELS_DIR, OUTPUTS_DIR
)
from collect_faces import collect_images_from_webcam
from train_model import train as train_model_fn

ensure_dirs()

st.set_page_config(
    page_title="ANN Face Recognition System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# NAVIGATION STATE (so buttons elsewhere on the page can jump to a tab)
# ===========================================================================
PAGES = ["🏠 Home", "➕ Register Person", "🎯 Train Model",
         "🔍 Face Recognition", "📊 Model Evaluation"]

if "nav" not in st.session_state:
    st.session_state.nav = PAGES[0]

def goto(page_name):
    st.session_state.nav = page_name
    st.session_state.nav_radio = page_name
    st.rerun()

# ===========================================================================
# GLOBAL STYLE — premium dark dashboard theme with Google Font + animations
# ===========================================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(1400px 700px at 8% -10%, #211238 0%, #120f22 40%, #0a0912 100%);
    }

    /* IMPORTANT: only hide the hamburger menu + footer watermark.
       Do NOT hide the whole <header>, because the sidebar's
       collapse/expand arrow lives inside it — hiding header
       makes the sidebar impossible to open/close. */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stToolbar"] { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="collapsedControl"] { visibility: visible !important; }

    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1280px; }

    .hero-wrap {
        padding: 46px 48px; border-radius: 28px; margin-bottom: 30px;
        background: linear-gradient(120deg, #6D28D9 0%, #7C3AED 30%, #4F46E5 65%, #2563EB 100%);
        background-size: 200% 200%;
        animation: gradientShift 10s ease infinite;
        box-shadow: 0 25px 60px rgba(109,40,217,0.4);
        position: relative; overflow: hidden;
        display:flex; justify-content:space-between; align-items:center; gap: 30px;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-wrap::after {
        content: ""; position: absolute; right: -80px; top: -80px;
        width: 280px; height: 280px; border-radius: 50%;
        background: rgba(255,255,255,0.07);
    }
    .hero-wrap::before {
        content: ""; position: absolute; left: 30%; bottom: -100px;
        width: 200px; height: 200px; border-radius: 50%;
        background: rgba(255,255,255,0.05);
    }
    .hero-eyebrow {
        display:inline-block; padding: 6px 16px; border-radius: 999px;
        background: rgba(255,255,255,0.18); color:#fff; font-size:0.76rem;
        font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
        margin-bottom: 16px; backdrop-filter: blur(6px);
        border: 1px solid rgba(255,255,255,0.25);
    }
    .hero-title {
        color:#fff; font-size:2.9rem; font-weight:900; margin:0; line-height:1.1;
        letter-spacing:-0.02em; position:relative; z-index:2;
        background: linear-gradient(90deg, #ffffff, #E9D5FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-sub   { color:rgba(255,255,255,0.92); font-size:1.08rem; margin-top:14px; max-width:620px; line-height:1.6; position:relative; z-index:2;}
    .hero-orb {
        width:130px; height:130px; border-radius:34px; flex-shrink:0;
        background: rgba(255,255,255,0.14); backdrop-filter: blur(10px);
        border:1px solid rgba(255,255,255,0.3); display:flex; align-items:center;
        justify-content:center; font-size:3.4rem; position:relative; z-index:2;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        animation: floaty 4s ease-in-out infinite;
    }
    @keyframes floaty {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .surface {
        background: linear-gradient(160deg, #17142A 0%, #120f20 100%);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 20px;
        padding: 24px 26px; height:100%; transition: all 0.25s ease;
    }
    .surface:hover { border-color: rgba(167,139,250,0.5); transform: translateY(-3px); box-shadow: 0 12px 30px rgba(109,40,217,0.18);}
    .surface h4 { color:#fff; margin: 8px 0 6px 0; font-size:1.08rem; font-weight:700; }
    .surface p  { color: #A5A0C2; font-size:0.92rem; margin:0; line-height:1.55; }
    .icon-badge {
        width:50px; height:50px; border-radius:15px; display:flex; align-items:center;
        justify-content:center; font-size:1.5rem; margin-bottom:8px;
    }

    .stat-card {
        background: linear-gradient(160deg, #17142A 0%, #120f20 100%);
        border:1px solid rgba(255,255,255,0.08); border-radius:18px;
        padding:20px 22px; text-align:left; transition: all 0.2s ease;
    }
    .stat-card:hover { border-color: rgba(167,139,250,0.4); transform: translateY(-2px); }
    .stat-num { color:#fff; font-size:2rem; font-weight:900; line-height:1; font-family:'JetBrains Mono', monospace;}
    .stat-label { color:#8B87A8; font-size:0.8rem; margin-top:8px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;}
    .stat-icon { font-size:1.3rem; float:right; opacity:0.7; }

    .pill {
        display:inline-block; padding:8px 18px; margin:5px 7px 5px 0; border-radius:999px;
        background: rgba(124,58,237,0.14); border:1px solid rgba(124,58,237,0.35);
        color:#C4B5FD; font-size:0.85rem; font-weight:600; transition: all 0.2s ease;
    }
    .pill:hover { background: rgba(124,58,237,0.28); }

    .flow-row { display:flex; align-items:center; gap:0; overflow-x:auto; padding: 8px 2px 16px 2px; }
    .flow-step {
        flex: 0 0 auto; min-width:160px; background: linear-gradient(160deg, #17142A, #120f20);
        border:1px solid rgba(255,255,255,0.1); border-radius:16px; padding:16px 18px; text-align:center;
        transition: all 0.2s ease;
    }
    .flow-step:hover { border-color: #A78BFA; transform: translateY(-2px); }
    .flow-step .icon { font-size: 1.5rem; margin-bottom: 4px; }
    .flow-step .n { color:#A78BFA; font-size:0.7rem; font-weight:800; letter-spacing:.1em; }
    .flow-step .t { color:#fff; font-size:0.88rem; font-weight:700; margin-top:4px; }
    .flow-arrow { color:#6D28D9; font-size:1.6rem; padding:0 12px; flex:0 0 auto; }

    .section-title {
        color:#fff; font-size:1.4rem; font-weight:800; margin: 8px 0 18px 0;
        display:flex; align-items:center; gap:10px; letter-spacing:-0.01em;
    }
    .section-sub { color:#8B87A8; font-size:0.9rem; margin-top:-12px; margin-bottom:18px; }

    .badge-ok   { background:rgba(34,197,94,0.15); color:#4ADE80; border:1px solid rgba(34,197,94,0.4); padding:5px 14px; border-radius:999px; font-size:0.8rem; font-weight:700;}
    .badge-warn { background:rgba(250,204,21,0.15); color:#FACC15; border:1px solid rgba(250,204,21,0.4); padding:5px 14px; border-radius:999px; font-size:0.8rem; font-weight:700;}
    .badge-live { background:rgba(248,113,113,0.15); color:#F87171; border:1px solid rgba(248,113,113,0.4); padding:4px 12px; border-radius:999px; font-size:0.75rem; font-weight:800; animation: pulse 1.4s infinite; }
    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }

    .result-ok {
        background: linear-gradient(90deg, rgba(34,197,94,0.18), rgba(34,197,94,0.03));
        border:1px solid rgba(34,197,94,0.4); border-radius:16px; padding:18px 22px; margin-top:12px;
    }
    .result-bad {
        background: linear-gradient(90deg, rgba(248,113,113,0.18), rgba(248,113,113,0.03));
        border:1px solid rgba(248,113,113,0.4); border-radius:16px; padding:18px 22px; margin-top:12px;
    }
    .result-name { color:#fff; font-size:1.4rem; font-weight:800; }
    .result-conf { color:#B9B6D0; font-size:0.92rem; margin-top:4px; }

    .avatar-row { display:flex; flex-wrap:wrap; gap:12px; margin-top:8px; }
    .avatar-chip {
        display:flex; align-items:center; gap:10px; background: linear-gradient(160deg, #17142A, #120f20);
        border:1px solid rgba(255,255,255,0.1); border-radius:14px; padding:10px 16px 10px 10px;
    }
    .avatar-circle {
        width:38px; height:38px; border-radius:50%; display:flex; align-items:center; justify-content:center;
        color:#fff; font-weight:800; font-size:0.95rem; flex-shrink:0;
    }
    .avatar-name { color:#fff; font-weight:700; font-size:0.9rem; }
    .avatar-count { color:#8B87A8; font-size:0.78rem; }

    section[data-testid="stSidebar"] { background:#0d0b18; border-right:1px solid rgba(255,255,255,0.06); min-width:300px !important; }
    section[data-testid="stSidebar"] .stRadio label { color:#D8D5EA; font-weight:600; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08); }

    .stButton>button {
        border-radius: 12px; font-weight: 700; border: none;
        background: linear-gradient(135deg, #7C3AED, #2563EB); color:#fff;
        padding: 0.65rem 1.3rem; transition: all 0.2s ease;
    }
    .stButton>button:hover { filter: brightness(1.15); transform: translateY(-1px); box-shadow: 0 8px 20px rgba(124,58,237,0.4); }

    .surface, .stat-card { color:#E7E5F5; }

    .stProgress > div > div > div { background: linear-gradient(90deg, #7C3AED, #2563EB); }
</style>
""", unsafe_allow_html=True)


AVATAR_COLORS = ["#7C3AED", "#2563EB", "#059669", "#D97706", "#DC2626", "#0891B2", "#DB2777"]
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def avatar_color(name):
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return AVATAR_COLORS[h % len(AVATAR_COLORS)]


def icon_badge(emoji, bg, fg):
    return f'<div class="icon-badge" style="background:{bg}; color:{fg};">{emoji}</div>'


def stat_card(value, label, icon=""):
    return f"""<div class="stat-card"><span class="stat-icon">{icon}</span>
    <div class="stat-num">{value}</div><div class="stat-label">{label}</div></div>"""


def draw_face_boxes(frame_bgr):
    """Run Haar-cascade face detection on a BGR frame and draw boxes live.
    Returns (annotated_frame, number_of_faces_found)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (237, 58, 124), 3)
        cv2.putText(frame_bgr, "Face Detected", (x, max(0, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (237, 58, 124), 2, cv2.LINE_AA)
    return frame_bgr, len(faces)


def confidence_gauge(confidence_pct, is_known):
    color = "#4ADE80" if is_known else "#F87171"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_pct,
        number={'suffix': "%", 'font': {'color': '#fff', 'size': 34, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#8B87A8', 'tickfont': {'color': '#8B87A8'}},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'threshold': {'line': {'color': "#fff", 'width': 2}, 'thickness': 0.75, 'value': 55},
        }
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={'color': "#fff"}
    )
    return fig


def people_bar_chart(people_list):
    names = people_list
    counts = [count_images(p) for p in people_list]
    colors = [avatar_color(p) for p in people_list]
    fig = go.Figure(go.Bar(
        x=counts, y=names, orientation='h', marker_color=colors,
        text=counts, textposition='outside', textfont={'color': '#fff'}
    ))
    fig.update_layout(
        height=max(220, 60 * len(names)), margin=dict(l=10, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={'gridcolor': 'rgba(255,255,255,0.08)', 'color': '#8B87A8'},
        yaxis={'color': '#E7E5F5', 'autorange': 'reversed'},
        font={'family': 'Inter'}
    )
    return fig


def metrics_bar_chart(results):
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    values = [results['accuracy']*100, results['precision']*100, results['recall']*100, results['f1']*100]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=["#7C3AED", "#2563EB", "#059669", "#D97706"],
        text=[f"{v:.1f}%" for v in values], textposition='outside', textfont={'color': '#fff'}
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=20, b=10), yaxis_range=[0, 110],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={'color': '#E7E5F5'}, yaxis={'gridcolor': 'rgba(255,255,255,0.08)', 'color': '#8B87A8'},
        font={'family': 'Inter'}
    )
    return fig


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown("### 🧠 &nbsp;ANN Face Recognition")
    st.caption("B.Tech AI/ML Project · v2.0")
    st.write("")

    if "nav_radio" not in st.session_state:
        st.session_state.nav_radio = st.session_state.nav

    page = st.radio(
        "Navigation", PAGES,
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.session_state.nav = page

    st.markdown("---")
    people = list_registered_people()
    _required_model_files = ["face_model.pkl", "label_encoder.pkl", "scaler.pkl", "pca.pkl"]
    model_ready = all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in _required_model_files)
    last_metrics = load_last_metrics()

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(stat_card(len(people), "People", "👥"), unsafe_allow_html=True)
    with sc2:
        st.markdown(stat_card(total_images(), "Images", "🖼️"), unsafe_allow_html=True)

    st.write("")
    status_badge = '<span class="badge-ok">✔ Model Trained</span>' if model_ready else '<span class="badge-warn">⚠ Not Trained</span>'
    st.markdown(f'<div style="margin:6px 0 16px 0;">{status_badge}</div>', unsafe_allow_html=True)

    if people:
        with st.expander("👥 Registered People", expanded=False):
            for p in people:
                c = avatar_color(p)
                st.markdown(f"""
                <div class="avatar-chip" style="margin-bottom:8px;">
                    <div class="avatar-circle" style="background:{c};">{p[0].upper()}</div>
                    <div><div class="avatar-name">{p}</div><div class="avatar-count">{count_images(p)} images</div></div>
                </div>""", unsafe_allow_html=True)


# ===========================================================================
# HOME
# ===========================================================================
if page == "🏠 Home":
    st.markdown(f"""
    <div class="hero-wrap">
        <div>
            <span class="hero-eyebrow">AI · Machine Learning · Computer Vision</span>
            <p class="hero-title">ANN Face Recognition<br>System</p>
            <p class="hero-sub">A complete, self-trained pipeline — face detection, preprocessing, PCA
            feature extraction and a genuine Artificial Neural Network classifier — wrapped in one
            interactive dashboard. Point a webcam at a face and watch it get caught in real time.</p>
        </div>
        <div class="hero-orb">🧠</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- credibility strip: badges that make it read like a real project ----
    st.markdown("""
    <div style="display:flex; flex-wrap:wrap; gap:10px; margin: -8px 0 26px 0;">
        <span class="pill" style="background:rgba(34,197,94,0.14); border-color:rgba(34,197,94,0.35); color:#86EFAC;">🎓 B.Tech AI/ML Major Project</span>
        <span class="pill">🧠 100% Self-Trained ANN (no pretrained models)</span>
        <span class="pill">⚡ Real-Time Face Detection</span>
        <span class="pill">📐 PCA Dimensionality Reduction</span>
    </div>
    """, unsafe_allow_html=True)

    acc_display = f"{last_metrics['accuracy']*100:.1f}%" if last_metrics else "—"
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(stat_card(len(people), "People Registered", "👥"), unsafe_allow_html=True)
    with c2: st.markdown(stat_card(total_images(), "Training Images", "🖼️"), unsafe_allow_html=True)
    with c3: st.markdown(stat_card(acc_display, "Last Model Accuracy", "🎯"), unsafe_allow_html=True)
    with c4: st.markdown(stat_card("Trained ✔" if model_ready else "Pending", "Model Status", "⚙️"), unsafe_allow_html=True)

    # ---- Quick start CTA row (working buttons) ----
    st.write("")
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        if st.button("➕ Register a New Person", width='stretch'):
            goto("➕ Register Person")
    with qc2:
        if st.button("🎯 Train the Model", width='stretch'):
            goto("🎯 Train Model")
    with qc3:
        if st.button("🔍 Start Live Recognition", width='stretch'):
            goto("🔍 Face Recognition")

    # ---- Objective / about section ----
    st.write("")
    ob1, ob2 = st.columns([1.3, 1])
    with ob1:
        st.markdown("""
        <div class="surface">
            <h4 style="font-size:1.25rem;">🎯 Project Objective</h4>
            <p style="font-size:0.98rem; line-height:1.7;">
            This system demonstrates an end-to-end face recognition pipeline built entirely
            from first principles — no third-party pretrained face-recognition APIs. A webcam
            feed is detected using <b style="color:#fff;">OpenCV Haar Cascades</b>, each face is
            cropped and normalized, reduced to its most informative features with
            <b style="color:#fff;">Principal Component Analysis (PCA)</b>, and finally classified
            by a hand-trained <b style="color:#fff;">Artificial Neural Network (MLP)</b> with
            ReLU-activated hidden layers and a Softmax output. Every stage — collection,
            training and evaluation — is fully interactive from this dashboard.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with ob2:
        st.markdown("""
        <div class="surface">
            <h4 style="font-size:1.25rem;">✨ Key Highlights</h4>
            <p style="font-size:0.95rem; line-height:1.9;">
            ✅ Real-time face box as soon as a face enters frame<br>
            ✅ Custom-trained ANN — fully explainable, no black-box API<br>
            ✅ Live accuracy, precision, recall, F1-score, confusion matrix<br>
            ✅ Per-person dataset tracking and management<br>
            ✅ One-click retraining as new people are added
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="surface">{icon_badge("🧩", "rgba(37,99,235,0.18)", "#60A5FA")}
            <h4>OpenCV Detection</h4>
            <p>Haar Cascade locates faces live from the webcam, drawing a box the instant a face enters frame — before anything else runs.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="surface">{icon_badge("🧠", "rgba(124,58,237,0.18)", "#A78BFA")}
            <h4>ANN / MLP Classifier</h4>
            <p>A self-trained feedforward neural network (2 hidden ReLU layers + Softmax) performs the actual recognition.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="surface">{icon_badge("📈", "rgba(34,197,94,0.18)", "#4ADE80")}
            <h4>Full Evaluation</h4>
            <p>Accuracy, precision, recall, F1-score and a confusion matrix are generated on every training run.</p>
        </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown('<p class="section-title">📌 Project Workflow</p>', unsafe_allow_html=True)
    steps = [("🎥", "01", "Webcam / Image"), ("🧩", "02", "Face Detection"),
             ("🎛️", "03", "Preprocessing"), ("📉", "04", "PCA Features"),
             ("🧠", "05", "ANN Classifier"), ("✅", "06", "Known / Unknown")]
    flow_html = '<div class="flow-row">'
    for i, (icon, num, label) in enumerate(steps):
        flow_html += f'<div class="flow-step"><div class="icon">{icon}</div><div class="n">STEP {num}</div><div class="t">{label}</div></div>'
        if i < len(steps) - 1:
            flow_html += '<div class="flow-arrow">→</div>'
    flow_html += '</div>'
    st.markdown(flow_html, unsafe_allow_html=True)

    if people:
        st.write("")
        st.markdown('<p class="section-title">👥 Dataset Overview</p>', unsafe_allow_html=True)
        st.plotly_chart(people_bar_chart(people), width='stretch', config={'displayModeBar': False})

    st.write("")
    st.markdown('<p class="section-title">🛠 Technology Stack</p>', unsafe_allow_html=True)
    techs = ["Python 3.x", "OpenCV", "NumPy", "Pandas", "Scikit-learn (MLPClassifier)",
             "PCA", "Matplotlib", "Seaborn", "Joblib", "Streamlit", "Plotly"]
    st.markdown("".join(f'<span class="pill">{t}</span>' for t in techs), unsafe_allow_html=True)

    st.write("")
    st.info("👈 Use the sidebar: **Register Person** → **Train Model** → **Face Recognition** → **Model Evaluation**")

    st.markdown("""
    <div style="text-align:center; margin-top:36px; padding-top:22px; border-top:1px solid rgba(255,255,255,0.08);">
        <p style="color:#6B6690; font-size:0.85rem;">
        Built with OpenCV, PCA & a custom Artificial Neural Network · Streamlit + Plotly dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# REGISTER PERSON  (live face-box preview while capturing)
# ===========================================================================
elif page == "➕ Register Person":
    st.markdown('<p class="section-title">➕ Register a New Person</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Capture face images from your webcam to build the training dataset. A box appears the instant a face is detected.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        person_name = st.text_input("Person's Name / ID", placeholder="e.g. Rahul")
        num_images = st.slider("Number of images to capture", 20, 150, 60, step=10)
        start_btn = st.button("📸 Start Capture", type="primary", width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="surface">
            <h4>💡 Tips for a good dataset</h4>
            <p>
            • Ensure good, even lighting on the face<br>
            • Slowly vary angle and expression during capture<br>
            • Keep only one face in frame at a time<br>
            • Aim for 50–100 images per person
            </p>
        </div>
        """, unsafe_allow_html=True)

    if start_btn:
        if not person_name.strip():
            st.error("Please enter a name before starting capture.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_placeholder = st.empty()

            def progress_cb(count, total):
                progress_bar.progress(count / total)
                status_text.markdown(f"**Captured {count}/{total} images**")

            def frame_cb(frame):
                # Live face-box overlay: the moment a face enters the frame,
                # a box + label is drawn on it before it's shown on screen.
                annotated, n_faces = draw_face_boxes(frame.copy())
                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB", width='stretch'
                )

            try:
                with st.spinner("Opening webcam..."):
                    saved = collect_images_from_webcam(
                        person_name.strip(), num_images=num_images,
                        progress_callback=progress_cb, frame_callback=frame_cb
                    )
                st.markdown(f"""
                <div class="result-ok">
                    <div class="result-name">✅ Saved {saved} images for "{person_name}"</div>
                    <div class="result-conf">Go to <b>Train Model</b> to (re)train the ANN with the updated dataset.</div>
                </div>""", unsafe_allow_html=True)
                st.balloons()
            except RuntimeError as e:
                st.error(f"❌ {e}")


# ===========================================================================
# TRAIN MODEL
# ===========================================================================
elif page == "🎯 Train Model":
    st.markdown('<p class="section-title">🎯 Train the ANN Model</p>', unsafe_allow_html=True)

    if len(people) < 2:
        st.warning("⚠️ You need at least **2 registered people** with images before training. "
                    "Go to 'Register Person' first.")
    else:
        avatar_html = '<div class="avatar-row">'
        for p in people:
            c = avatar_color(p)
            avatar_html += f"""<div class="avatar-chip">
                <div class="avatar-circle" style="background:{c};">{p[0].upper()}</div>
                <div><div class="avatar-name">{p}</div><div class="avatar-count">{count_images(p)} images</div></div>
            </div>"""
        avatar_html += '</div>'
        st.markdown(f'<div class="surface" style="margin-bottom:18px;"><p style="margin-bottom:6px;">Ready to train on <b style="color:#fff;">{len(people)}</b> registered people:</p>{avatar_html}</div>', unsafe_allow_html=True)

        if st.button("🚀 Start Training", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_cb(msg, pct):
                progress_bar.progress(pct)
                status_text.markdown(f"**{msg}**")

            try:
                with st.spinner("Training in progress..."):
                    results = train_model_fn(progress_callback=progress_cb)
                st.success("✅ Training complete!")

                left, right = st.columns([1, 1])
                with left:
                    st.plotly_chart(metrics_bar_chart(results), width='stretch', config={'displayModeBar': False})
                with right:
                    st.markdown(f"""
                    <div class="surface">
                        <h4>📋 Training Summary</h4>
                        <p>
                        Classes: <b style="color:#fff;">{results['n_classes']}</b> ({', '.join(results['classes'])})<br>
                        Training samples: <b style="color:#fff;">{results['n_train']}</b><br>
                        Testing samples: <b style="color:#fff;">{results['n_test']}</b><br>
                        Epochs run: <b style="color:#fff;">{results['n_iterations_run']}</b>
                        </p>
                    </div>""", unsafe_allow_html=True)

                if results.get("dataset_warnings"):
                    st.markdown("""<div class="surface" style="border-color:rgba(250,204,21,0.4); margin-bottom:16px;">
                        <h4 style="color:#FACC15;">⚠️ Dataset Quality Warnings</h4>
                        <p>""" + "<br>".join(f"• {w}" for w in results["dataset_warnings"]) + """</p>
                        <p style="margin-top:10px; color:#8B87A8;">More, better-balanced images per person will directly improve live recognition accuracy.</p>
                    </div>""", unsafe_allow_html=True)

                with st.expander("📄 Detailed Classification Report"):
                    st.code(results["report"])

                col1, col2 = st.columns(2)
                with col1:
                    st.image(results["confusion_matrix_path"], caption="Confusion Matrix")
                with col2:
                    st.image(results["training_graph_path"], caption="Training Curve")

            except RuntimeError as e:
                st.error(f"❌ {e}")


# ===========================================================================
# FACE RECOGNITION  (Upload / Snapshot / Live real-time)
# ===========================================================================
elif page == "🔍 Face Recognition":
    st.markdown('<p class="section-title">🔍 Face Recognition</p>', unsafe_allow_html=True)

    if not model_ready:
        st.warning("⚠️ Model not trained yet. Please go to 'Train Model' first.")
    else:
        from recognize import FaceRecognizer
        recognizer = FaceRecognizer()

        mode = st.radio(
            "Choose input method:",
            ["Live Webcam (Real-time)", "Upload Image", "Webcam Snapshot"],
            horizontal=True
        )

        def show_results(results):
            if len(results) == 0:
                st.warning("No face detected.")
                return
            for name, confidence, box in results:
                is_known = name != "Unknown"
                r1, r2 = st.columns([1.4, 1])
                with r1:
                    if is_known:
                        c = avatar_color(name)
                        st.markdown(f"""
                        <div class="result-ok">
                            <div style="display:flex; align-items:center; gap:14px;">
                                <div class="avatar-circle" style="background:{c}; width:52px; height:52px; font-size:1.3rem;">{name[0].upper()}</div>
                                <div>
                                    <div class="result-name">✅ {name}</div>
                                    <div class="result-conf">Recognized with {confidence*100:.1f}% confidence</div>
                                </div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-bad">
                            <div class="result-name">❌ Unknown Person</div>
                            <div class="result-conf">Confidence {confidence*100:.1f}% — below recognition threshold</div>
                        </div>""", unsafe_allow_html=True)
                with r2:
                    st.plotly_chart(confidence_gauge(confidence*100, is_known),
                                     width='stretch', config={'displayModeBar': False})

        # ------------------------------------------------------------
        # LIVE WEBCAM — continuously scans; catches a face the moment
        # it's confidently recognized (or after a short timeout).
        # ------------------------------------------------------------
        if mode == "Live Webcam (Real-time)":
            st.markdown(
                '<p class="section-sub">Click Start, look at the camera — recognition happens automatically in real time.</p>',
                unsafe_allow_html=True
            )
            start_live = st.button("▶️ Start Live Scan", type="primary")
            live_placeholder = st.empty()
            info_placeholder = st.empty()

            if start_live:
                # On Windows the default backend (MSMF) can hang or fail to
                # open instantly — CAP_DSHOW is far more reliable there.
                import platform
                if platform.system() == "Windows":
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(0)

                # Warm up: give the camera driver a moment and a few
                # throwaway reads before trusting isOpened()/read().
                opened = cap.isOpened()
                if opened:
                    for _ in range(5):
                        ok, _ = cap.read()
                        if ok:
                            break
                        time.sleep(0.1)
                    else:
                        opened = False

                if not opened:
                    cap.release()
                    st.error(
                        "❌ Could not access the webcam.\n\n"
                        "Checklist:\n"
                        "- Close any other app using the camera (Zoom, Teams, browser tab, etc.)\n"
                        "- Check Windows camera privacy permission for Python/your terminal\n"
                        "- Make sure no other part of this app has the camera open at the same time\n"
                        "- Try a different camera index if you have more than one webcam"
                    )
                else:
                    info_placeholder.markdown('<span class="badge-live">● LIVE — scanning…</span>', unsafe_allow_html=True)
                    status_placeholder = st.empty()
                    CONFIDENCE_THRESHOLD = 0.55
                    MAX_FRAMES = 150          # ~ safety timeout (about 15–20s)
                    HOLD_FRAMES_NEEDED = 4    # must see the same confident face a few frames in a row
                    hold_count = 0
                    last_name = None
                    found = False
                    error_msg = None
                    try:
                        for frame_i in range(MAX_FRAMES):
                            ret, frame = cap.read()
                            if not ret:
                                error_msg = "Webcam stopped sending frames mid-scan."
                                break

                            try:
                                annotated, results = recognizer.recognize_frame(frame)
                            except Exception as e:
                                error_msg = f"Recognition error: {e}"
                                break

                            live_placeholder.image(
                                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                channels="RGB", width='stretch'
                            )
                            status_placeholder.caption(f"Frame {frame_i + 1}/{MAX_FRAMES} · faces seen this frame: {len(results)}")

                            if results:
                                name, confidence, box = results[0]
                                if name != "Unknown" and confidence >= CONFIDENCE_THRESHOLD:
                                    if name == last_name:
                                        hold_count += 1
                                    else:
                                        last_name = name
                                        hold_count = 1
                                    if hold_count >= HOLD_FRAMES_NEEDED:
                                        found = True
                                        final_results = results
                                        break
                                else:
                                    hold_count = 0
                                    last_name = None

                            time.sleep(0.03)
                    finally:
                        cap.release()

                    info_placeholder.empty()
                    status_placeholder.empty()
                    if error_msg:
                        st.error(f"❌ {error_msg}")
                    elif found:
                        st.success("🎯 Face caught and recognized!")
                        show_results(final_results)
                    else:
                        st.warning("⏱️ Scan timed out — no confidently recognized face held steady. Try again with better lighting / closer to the camera.")

        elif mode == "Upload Image":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                annotated, results = recognizer.recognize_frame(frame)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width='stretch')
                show_results(results)

        else:  # Webcam Snapshot
            snap = st.camera_input("Take a photo")
            if snap is not None:
                image = Image.open(snap).convert("RGB")
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                annotated, results = recognizer.recognize_frame(frame)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width='stretch')
                show_results(results)


# ===========================================================================
# MODEL EVALUATION
# ===========================================================================
elif page == "📊 Model Evaluation":
    st.markdown('<p class="section-title">📊 Model Evaluation</p>', unsafe_allow_html=True)

    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    graph_path = os.path.join(OUTPUTS_DIR, "training_graph.png")

    if not model_ready or not os.path.exists(cm_path):
        st.warning("⚠️ No evaluation results yet. Please train the model first.")
    else:
        if last_metrics:
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(stat_card(f"{last_metrics['accuracy']*100:.1f}%", "Accuracy", "🎯"), unsafe_allow_html=True)
            with c2: st.markdown(stat_card(f"{last_metrics['precision']*100:.1f}%", "Precision", "🔍"), unsafe_allow_html=True)
            with c3: st.markdown(stat_card(f"{last_metrics['recall']*100:.1f}%", "Recall", "📡"), unsafe_allow_html=True)
            with c4: st.markdown(stat_card(f"{last_metrics['f1']*100:.1f}%", "F1-score", "⚖️"), unsafe_allow_html=True)
            st.caption(f"Last trained: {last_metrics.get('trained_at', 'unknown')} · "
                       f"{last_metrics['n_classes']} classes · "
                       f"{last_metrics['n_train']} train / {last_metrics['n_test']} test samples")
            st.write("")

        col1, col2 = st.columns(2)
        with col1:
            st.image(cm_path, caption="Confusion Matrix", width='stretch')
        with col2:
            st.image(graph_path, caption="Training Curve (Loss / Validation Accuracy)",
                      width='stretch')

        st.info("Re-run training from the 'Train Model' tab any time you add new people "
                "or more images — accuracy, precision, recall and F1-score are recomputed "
                "on every run.")