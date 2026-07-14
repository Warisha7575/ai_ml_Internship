from src.utils import (
    calculate_ear,
    calculate_fatigue_score,
    get_recommendation,
    LEFT_EYE,
    RIGHT_EYE,
    get_eye_landmarks
)
import streamlit as st
import cv2
import mediapipe as mp
import time
import joblib
from src.alarm import start_alarm, stop_alarm

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fatigue_model.pkl"
)

# PAGE CONFIG
st.set_page_config(
    page_title="DriverGuard AI",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.metric-card {
    background-color: #1E2635;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.big-title {
    font-size: 42px;
    font-weight: bold;
    background: linear-gradient(90deg, #4ECDC4, #44A08D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    color: #94A3B8;
    margin-bottom: 20px;
}
.stButton>button {
    background: linear-gradient(90deg, #4ECDC4, #44A08D);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    padding: 0.75rem 1.5rem;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #44A08D, #4ECDC4);
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🚗 DriverGuard AI")
st.sidebar.markdown("---")
st.sidebar.success("🟢 System Ready")

# Load model with error handling
try:
    if not os.path.exists(MODEL_PATH):
        st.sidebar.error("❌ Model file not found!")
        st.stop()
    model = joblib.load(MODEL_PATH)
    st.sidebar.info("✅ Fatigue Model Loaded")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.write("Version 1.0")
st.sidebar.write("AI Internship Project")

st.markdown(
    '<div class="big-title">🚗 DriverGuard AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Real-Time Driver Fatigue Monitoring System</div>',
    unsafe_allow_html=True
)
st.divider()

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

if st.button("▶ Start Monitoring"):
    # Initialize camera with error handling
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open camera! Please check camera permissions.")
        st.stop()

    # Main dashboard row
    left_col, right_col = st.columns([2, 1])

    with left_col:
        frame_placeholder = st.empty()

    with right_col:
        status_placeholder = st.empty()
        st.markdown("<br>", unsafe_allow_html=True)
        fatigue_placeholder = st.empty()
        st.markdown("<br>", unsafe_allow_html=True)
        confidence_placeholder = st.empty()

    st.divider()

    e1, e2, e3, e4 = st.columns(4)
    ear_placeholder = e1.empty()
    blink_placeholder = e2.empty()
    closure_placeholder = e3.empty()
    perclos_placeholder = e4.empty()

    recommendation_placeholder = st.empty()

    blink_count = 0
    closed_frames = 0
    total_frames = 0
    closed_eye_frames = 0

    EAR_THRESHOLD = 0.20

    eye_closed_start = None
    session_start = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        total_frames += 1
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
        results = face_mesh.process(rgb)
        status = "FOCUSED"
        fatigue_score = 0
        confidence = 100
        ear = 0
        blink_rate = 0
        closure_duration = 0
        perclos = 0

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            h, w, _ = frame.shape

            # Get both eyes
            left_eye_points = get_eye_landmarks(face, LEFT_EYE, (h, w))
            right_eye_points = get_eye_landmarks(face, RIGHT_EYE, (h, w))

            # Draw eye landmarks
            for (x, y) in left_eye_points + right_eye_points:
                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 255, 0),
                    -1
                )

            # Calculate EAR as average of both eyes
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            ear = (left_ear + right_ear) / 2

            if ear < EAR_THRESHOLD:
                closed_frames += 1
                closed_eye_frames += 1
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                closure_duration = (time.time() - eye_closed_start)
            else:
                if closed_frames >= 2:
                    blink_count += 1
                closed_frames = 0
                eye_closed_start = None

            elapsed_minutes = (time.time() - session_start) / 60
            if elapsed_minutes > 0:
                blink_rate = blink_count / elapsed_minutes
            perclos = (closed_eye_frames / total_frames) * 100

            features = [[
                ear,
                blink_rate,
                closure_duration,
                perclos
            ]]

            try:
                prediction = model.predict(features)[0]
                probability = max(model.predict_proba(features)[0])
                confidence = round(probability * 100, 2)
                status = "DROWSY" if prediction == 1 else "FOCUSED"
                if status == "DROWSY":
                    start_alarm()
                else:
                    stop_alarm()
            except Exception as e:
                confidence = 0
                status = "FOCUSED"

            fatigue_score = calculate_fatigue_score(
                ear,
                perclos,
                closure_duration
            )
            recommendation = get_recommendation(
                fatigue_score
            )

            # ------------------------
            # STATUS
            # ------------------------
            if status == "FOCUSED":
                status_placeholder.markdown(
                    f"""
                    <div style="
                        background:#163f25;
                        padding:20px;
                        border-radius:12px;
                        text-align:center;
                        color:white;
                        font-size:28px;
                        font-weight:bold;">
                        🟢 {status}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                status_placeholder.markdown(
                    f"""
                    <div style="
                        background:#5a1414;
                        padding:20px;
                        border-radius:12px;
                        text-align:center;
                        color:white;
                        font-size:28px;
                        font-weight:bold;">
                        🔴 {status}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            fatigue_placeholder.metric(
                "Fatigue Score",
                f"{fatigue_score}%"
            )
            confidence_placeholder.metric(
                "Confidence",
                f"{confidence}%"
            )
            ear_placeholder.metric(
                "EAR",
                f"{ear:.2f}"
            )
            blink_placeholder.metric(
                "Blink Rate",
                f"{blink_rate:.1f}/min"
            )
            closure_placeholder.metric(
                "Eye Closure",
                f"{closure_duration:.2f}s"
            )
            perclos_placeholder.metric(
                "PERCLOS",
                f"{perclos:.1f}%"
            )

            if status == "DROWSY":
                recommendation_placeholder.error(f"⚠️ {recommendation}")
            elif fatigue_score >= 30:
                recommendation_placeholder.warning(f"⚠️ {recommendation}")
            else:
                recommendation_placeholder.success(f"✅ {recommendation}")

            cv2.putText(
                frame,
                f"{status}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if status == "FOCUSED" else (0, 0, 255),
                2
            )

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
        frame_placeholder.image(
            frame_rgb,
            channels="RGB",
            use_container_width=700
        )

    cap.release()
    stop_alarm()  # Ensure alarm stops when monitoring ends
