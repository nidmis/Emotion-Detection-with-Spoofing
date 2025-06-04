import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras  
import tempfile
import os
from PIL import Image
import plotly.express as px
from datetime import datetime
import pandas as pd
 
APP_VERSION = "1.1.0"
 
st.set_page_config(
    page_title="AI Facial Insights Platform",  
    page_icon="✨",  
    layout="wide",
    initial_sidebar_state="expanded"
)
 
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .result-card { 
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .success-card { 
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white; 
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .warning-card { 
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #333; 
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2193b0 0%, #6dd5ed 100%);
    }
    
    .live-results { 
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .result-summary-area {
        margin-top: 30px; 
        padding: 20px;
        background-color: #f8f9fa; 
        border-radius: 12px; 
        border: 1px solid #e9ecef;
    }
    .result-summary-area h3.summary-title { 
        color: #0056b3;
        margin-bottom: 20px;
        text-align: center;
        font-size: 1.5em;
    }
    .face-result-item {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-left: 7px solid #007bff; 
        border-radius: 8px;
        padding: 20px; 
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08); 
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    .face-result-item:hover {
        transform: translateY(-5px); 
        box-shadow: 0 6px 15px rgba(0,0,0,0.12);
    }
    .face-result-item h4.face-id-title {  
        margin-top: 0;
        margin-bottom: 15px; 
        color: #0056b3; 
        font-size: 1.2em; 
        border-bottom: 2px solid #f0f2f6; 
        padding-bottom: 10px;
    }
    .result-label {
        font-weight: bold;
        color: #343a40; 
        margin-right: 8px;
        font-size: 1.05em;
    }
    .result-value-emotion {
        font-weight: bold;
        font-size: 1.05em;
    }
    .result-value-spoof-real {
        color: #198754; 
        font-weight: bold;
        font-size: 1.05em;
    }
    .result-value-spoof-fake {
        color: #dc3545; 
        font-weight: bold;
        font-size: 1.05em;
    }
    .confidence-score {
        font-size: 0.95em; 
        color: #495057; 
    }

</style>
""", unsafe_allow_html=True)
 
if 'analysis_results' not in st.session_state: 
    st.session_state.analysis_results = []
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False
if 'live_cam_face_results' not in st.session_state: 
    st.session_state.live_cam_face_results = []
if 'spoof_real_threshold_ss' not in st.session_state:
    st.session_state.spoof_real_threshold_ss = 0.70
if 'camera_permission' not in st.session_state:
    st.session_state.camera_permission = False
if 'camera_frame' not in st.session_state:
    st.session_state.camera_frame = None
 
@st.cache_resource
def load_emotion_model():
    """Load the emotion detection model"""
    try:
        model_path = 'fer2013_emotion_model_augmented_best.h5'
        if not os.path.exists(model_path):
            st.error(f"Emotion model file not found at: {model_path}")
            return None
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading emotion model: {str(e)}")
        return None

@st.cache_resource
def load_spoof_model():
    """Load the spoof detection model"""
    try:
        model_path = 'spoof_detection_mobilenet_finetuned_best.h5'
        if not os.path.exists(model_path):
            st.error(f"Spoof model file not found at: {model_path}")
            return None
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading spoof detection model: {str(e)}")
        return None

@st.cache_resource
def load_face_cascade():
    """Load face detection cascade"""
    cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        st.error(f"Haar Cascade file not found at: {cascade_path}. Please download it and place it in the script's directory.")
        try:  
            return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e_cv2_path:
            st.error(f"Failed to load Haar Cascade from cv2.data path: {e_cv2_path}")
            return None
    return cv2.CascadeClassifier(cascade_path)


EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_face_for_emotion(face):
    """Preprocess face for emotion detection"""
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face

def preprocess_face_for_spoof(face):
    """Preprocess face for spoof detection"""
    face = cv2.resize(face, (160, 160)) 
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def detect_faces(image, face_cascade):
    """Detect faces in the image"""
    if face_cascade is None:
        st.error("Face cascade not loaded. Cannot detect faces.")
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)) 
    return faces

def analyze_face(face_roi, emotion_model, spoof_model, spoof_threshold):
    """Analyze a single face for emotion and spoof detection with dynamic threshold"""
    results = {}
    
    if face_roi is None or face_roi.size == 0:
        results['emotion'] = "Invalid ROI"
        results['emotion_confidence'] = 0
        results['is_real'] = False 
        results['real_confidence'] = 0
        results['display_spoof_confidence'] = 1.0 
        return results

    if emotion_model is not None:
        try:
            emotion_face = preprocess_face_for_emotion(face_roi)
            emotion_predictions = emotion_model.predict(emotion_face, verbose=0)
            emotion_probabilities = emotion_predictions[0]
            predicted_emotion = EMOTION_LABELS[np.argmax(emotion_probabilities)]
            emotion_confidence = np.max(emotion_probabilities)
            
            results['emotion'] = predicted_emotion
            results['emotion_confidence'] = emotion_confidence
            results['emotion_probabilities'] = dict(zip(EMOTION_LABELS, emotion_probabilities))
        except Exception as e:
            results['emotion'] = "Error"
            results['emotion_confidence'] = 0
            st.warning(f"Emotion detection error: {e}")
    
    if spoof_model is not None:
        try:
            spoof_face = preprocess_face_for_spoof(face_roi)
            spoof_predictions = spoof_model.predict(spoof_face, verbose=0)
            real_probability = spoof_predictions[0][0] 
            
            is_real = real_probability >= spoof_threshold 
            results['is_real'] = is_real
            results['real_confidence'] = real_probability 
            results['display_spoof_confidence'] = real_probability if is_real else (1 - real_probability)

        except Exception as e:
            results['is_real'] = False 
            results['real_confidence'] = 0
            results['display_spoof_confidence'] = 1.0 
            st.warning(f"Spoof detection error: {e}")
    
    return results

def create_results_display(results):
    """Create a formatted display of analysis results using user's card styles"""
    col1, col2 = st.columns(2)
    
    with col1:
        if 'emotion' in results and results['emotion'] != "Error" and results['emotion'] != "Invalid ROI":
            emotion_card_class = "success-card" if results.get('emotion_confidence', 0) > 0.6 else "warning-card"
            st.markdown(f"""
            <div class="{emotion_card_class}">
                <h3>🎭 Emotion Analysis</h3>
                <h2>{results['emotion']}</h2>
                <p>Confidence: {results.get('emotion_confidence', 0):.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        elif 'emotion' in results:
             st.markdown(f"""
            <div class="warning-card">
                <h3>🎭 Emotion Analysis</h3>
                <h2>{results['emotion']}</h2>
                <p>Confidence: N/A</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        if 'is_real' in results:
            auth_card_class = "success-card" if results['is_real'] else "warning-card"
            real_status = "AUTHENTIC" if results['is_real'] else "SPOOFED"
            confidence_to_display = results.get('display_spoof_confidence', 0)
            
            st.markdown(f"""
            <div class="{auth_card_class}">
                <h3>🛡️ Authenticity Verification</h3>
                <h2>{real_status}</h2>
                <p>Confidence: {confidence_to_display:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

def create_emotion_chart(emotion_probabilities):
    """Create emotion probability chart"""
    if not emotion_probabilities or not isinstance(emotion_probabilities, dict):
        return None
    df = pd.DataFrame(list(emotion_probabilities.items()), columns=['Emotion', 'Probability'])
    fig = px.bar(df, x='Emotion', y='Probability', 
                 title="Emotion Detection Probabilities",
                 color='Probability',
                 color_continuous_scale='viridis')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333')  
    )
    return fig

def draw_detections_on_frame(frame, x, y, w, h, emotion, emotion_conf, spoof_status, spoof_conf_display):
    """Draws emotion and spoof detections on the frame."""
    emotion_color_map = {
        'Happy': (0, 255, 0), 'Surprise': (0, 255, 255), 'Neutral': (200, 200, 200),
        'Sad': (255, 0, 0), 'Angry': (0, 0, 255), 'Fear': (128, 0, 128), 'Disgust': (0, 128, 0)
    }
    rect_color = emotion_color_map.get(emotion, (255, 255, 255)) 
    cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, 2)

    emotion_text = f"{emotion} ({emotion_conf:.0f}%)"
    text_y_offset = y - 10
    (text_width, text_height), _ = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x, text_y_offset - text_height - 5), (x + text_width + 5, text_y_offset + 5), rect_color, -1)
    text_color_on_rect = (0,0,0) if sum(rect_color[:3]) > 382 else (255,255,255) 
    cv2.putText(frame, emotion_text, (x + 2, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color_on_rect, 1, cv2.LINE_AA)

    spoof_text_color_on_img = (0, 255, 0) if spoof_status == "Real" else (0, 0, 255)
    spoof_text_on_img = f"{spoof_status} ({spoof_conf_display:.0f}%)"
    text_y_offset_spoof = y + h + 18
    cv2.putText(frame, spoof_text_on_img, (x, text_y_offset_spoof), cv2.FONT_HERSHEY_SIMPLEX, 0.45, spoof_text_color_on_img, 1, cv2.LINE_AA)

def main():
    st.markdown("""
    <div class="main-header">
        <h1>✨ AI Facial Insights Platform</h1> <!-- Updated App Name -->
        <p>Advanced Emotion Detection and Authenticity Verification</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("## 🧭 Navigation Panel")  
     
    nav_options = {
        "Live Webcam Analysis": "📹",
        "Image Upload Analysis": "🖼️",
        "Video File Analysis": "🎬",
        "Analysis Dashboard": "📊"
    }
    analysis_mode_display = st.sidebar.selectbox(
        "Select Analysis Mode",
        list(nav_options.keys()),
        format_func=lambda option: f"{nav_options[option]} {option}" # Display option with icon
    )
    
    emotion_model_loaded = load_emotion_model()
    spoof_model_loaded = load_spoof_model()
    face_cascade_loaded = load_face_cascade()

    if not all([emotion_model_loaded, spoof_model_loaded, face_cascade_loaded]):
        st.error("⚠️ Critical components (Models/Haar Cascade) failed to load. Application cannot proceed. Please ensure model files (`.h5`) and `haarcascade_frontalface_default.xml` are in the script's directory.")
        return 
    
    if analysis_mode_display == "Live Webcam Analysis":
        webcam_analysis(emotion_model_loaded, spoof_model_loaded, face_cascade_loaded)
    elif analysis_mode_display == "Image Upload Analysis":
        image_analysis(emotion_model_loaded, spoof_model_loaded, face_cascade_loaded)
    elif analysis_mode_display == "Video File Analysis":
        video_analysis(emotion_model_loaded, spoof_model_loaded, face_cascade_loaded)
    elif analysis_mode_display == "Analysis Dashboard":
        analysis_dashboard()
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"App Version: {APP_VERSION}")

def webcam_analysis(emotion_model, spoof_model, face_cascade):
    st.markdown("## 📹 Live Webcam Facial Analysis")
    st.markdown("Real-time emotion detection and authenticity verification from your webcam feed.")
    
    # Camera permission handling
    if not st.session_state.camera_permission:
        st.warning("🔒 Camera access requires your permission. Click below to enable camera.")
        if st.button("Allow Camera Access", key="request_camera_permission"):
            st.session_state.camera_permission = True
            st.rerun()
        return
    
    st.session_state.spoof_real_threshold_ss = st.slider(
        "Spoof Detection 'Real' Threshold:",
        min_value=0.0, max_value=1.0,
        value=st.session_state.spoof_real_threshold_ss,
        step=0.01, format="%.2f",
        help="Adjust confidence for 'Real' classification. Higher is stricter."
    )
    current_spoof_threshold = st.session_state.spoof_real_threshold_ss
    
    st.info(
        f"""**Note on Spoof Detection**: System classifies as 'Real' if confidence ≥ {current_spoof_threshold*100:.0f}%. 
        Accuracy depends on the model. Experiment with the threshold.
        """, icon="⚠️"
    )
    st.markdown("---")

    # Use Streamlit's camera_input to trigger browser permission
    camera_img = st.camera_input("Look at the camera", key="camera_feed")
    
    if camera_img is not None:
        # Convert camera image to OpenCV format
        bytes_data = camera_img.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        st.session_state.camera_frame = cv2_img

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Start Analysis", key="start_analysis", use_container_width=True):
            st.session_state.run_camera = True
            st.session_state.live_cam_face_results = [] 
    with col_btn2:
        if st.button("🛑 Stop Analysis", key="stop_analysis", use_container_width=True):
            st.session_state.run_camera = False 

    image_placeholder = st.empty()
    results_display_area = st.empty() 

    if st.session_state.run_camera and st.session_state.camera_frame is not None:
        frame = st.session_state.camera_frame.copy()
        faces = detect_faces(frame, face_cascade)
        
        current_frame_results_temp = []
        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    analysis_result = analyze_face(face_roi, emotion_model, spoof_model, current_spoof_threshold)
                    analysis_result['id'] = i + 1 
                    current_frame_results_temp.append(analysis_result)
                    
                    draw_detections_on_frame(
                        frame, x, y, w, h,
                        analysis_result.get('emotion', 'N/A'),
                        analysis_result.get('emotion_confidence', 0) * 100, 
                        "Real" if analysis_result.get('is_real', False) else "Spoof",
                        analysis_result.get('display_spoof_confidence', 0) * 100 
                    )
        
        st.session_state.live_cam_face_results = current_frame_results_temp
        image_placeholder.image(frame, channels="BGR", use_container_width=True, caption="Analysis Result")

    if st.session_state.live_cam_face_results:
        with results_display_area.container():
            st.markdown('<div class="result-summary-area">', unsafe_allow_html=True)
            st.markdown(f"<h3 class='summary-title'>Analysis Summary ({len(st.session_state.live_cam_face_results)} Face(s))</h3>", unsafe_allow_html=True)
            
            for i, res_item in enumerate(st.session_state.live_cam_face_results):
                st.markdown(f"<h4 class='face-id-title'>👤 Face {res_item.get('id', i+1)}</h4>", unsafe_allow_html=True)
                create_results_display(res_item) 
                
                if 'emotion_probabilities' in res_item and res_item['emotion_probabilities']:
                    with st.expander(f"Emotion Details for Face {res_item.get('id', i+1)}", expanded=False):
                        fig = create_emotion_chart(res_item['emotion_probabilities'])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("💾 Save Current Analysis to Dashboard", key="save_live_analysis", use_container_width=True):
                for res_to_save in st.session_state.live_cam_face_results:
                    save_data = res_to_save.copy() 
                    save_data['timestamp'] = datetime.now()
                    save_data['source'] = 'live_capture_saved'
                    st.session_state.analysis_results.append(save_data)
                st.success(f"✅ {len(st.session_state.live_cam_face_results)} face analysis result(s) saved to dashboard!")

    elif not st.session_state.run_camera: 
        image_placeholder.info("Analysis is off. Click 'Start Analysis' to begin.")
        results_display_area.empty()

def image_analysis(emotion_model, spoof_model, face_cascade):
    st.markdown("## 🖼️ Image Upload Analysis")
    st.markdown("Upload an image for comprehensive facial analysis including emotion detection and authenticity verification.")
    
    uploaded_file = st.file_uploader(
        "Select Image File",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image containing faces for analysis"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True) 
        
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
 
        image_spoof_threshold = st.slider(
            "Spoof Threshold for this Image:", 0.0, 1.0, 
            st.session_state.spoof_real_threshold_ss,  
            0.01, "%.2f", key="image_spoof_thresh"
        )

        if st.button("🔍 Perform Facial Analysis", key="analyze_image", use_container_width=True):
            with st.spinner("Analyzing facial features..."):
                faces = detect_faces(opencv_image, face_cascade)
                
                if len(faces) == 0:
                    st.warning("⚠️ No faces detected in the uploaded image.")
                    return
                
                st.success(f"✅ Detected {len(faces)} face(s) in the image")
                
                for i, (x, y, w, h) in enumerate(faces):
                    st.markdown(f"### 👤 Face {i+1} Analysis Results")
                    face_roi = opencv_image[y:y+h, x:x+w]
                    
                    if face_roi.size > 0:
                        results = analyze_face(face_roi, emotion_model, spoof_model, image_spoof_threshold) 
                        create_results_display(results)
                        
                        results['timestamp'] = datetime.now()
                        results['source'] = 'image_upload'
                        st.session_state.analysis_results.append(results)
                        
                        if 'emotion_probabilities' in results and results['emotion_probabilities']:
                            fig = create_emotion_chart(results['emotion_probabilities'])
                            if fig: st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Face {i+1}: ROI is empty, skipping analysis.")

def video_analysis(emotion_model, spoof_model, face_cascade):
    st.markdown("## 🎬 Video File Analysis")
    st.markdown("Upload a video file for frame-by-frame facial analysis.")
    
    uploaded_video = st.file_uploader(
        "Select Video File",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file for facial analysis"
    )
    
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        col1_vid_opts, col2_vid_opts, col3_vid_opts = st.columns(3)
        with col1_vid_opts:
            frame_skip = st.slider("Frame Skip Interval", 1, 30, 5, 
                                   help="Analyze every Nth frame to optimize processing", key="video_frame_skip")
        with col2_vid_opts:
            max_frames_to_process = st.slider("Maximum Frames to Process", 10, 500, 100, 
                                       help="Limit processing for large videos", key="video_max_frames")
        with col3_vid_opts:
            video_spoof_threshold = st.slider(
                "Spoof Threshold for this Video:", 0.0, 1.0, 
                st.session_state.spoof_real_threshold_ss,  
                0.01, "%.2f", key="video_spoof_thresh"
            )

        if st.button("🎯 Start Video Analysis", key="analyze_video", use_container_width=True):
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0 
            processed_frame_count = 0 
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            video_results_list = [] 
            
            try:
                total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_video_frames > 0:
                     effective_max_frames = min(max_frames_to_process, total_video_frames // frame_skip if frame_skip > 0 else total_video_frames)
                else:
                    effective_max_frames = max_frames_to_process 
            except:
                total_video_frames = 0 
                effective_max_frames = max_frames_to_process

            status_text.text(f"Preparing to process video... Target frames: {effective_max_frames}")

            while cap.isOpened() and processed_frame_count < max_frames_to_process:
                ret, frame = cap.read()
                if not ret: break
                
                if frame_idx % frame_skip == 0:
                    faces = detect_faces(frame, face_cascade)
                    
                    if len(faces) > 0:
                        for x, y, w, h in faces:
                            face_roi = frame[y:y+h, x:x+w]
                            if face_roi.size > 0:
                                results = analyze_face(face_roi, emotion_model, spoof_model, video_spoof_threshold)
                                results['frame'] = frame_idx
                                results['timestamp'] = datetime.now() 
                                results['source'] = 'video_file'
                                video_results_list.append(results)
                    
                    processed_frame_count += 1
                    if effective_max_frames > 0:
                        progress = processed_frame_count / effective_max_frames
                        progress_bar.progress(min(progress, 1.0)) 
                    status_text.text(f"Processing video frame {frame_idx}... Analyzed {processed_frame_count}/{effective_max_frames} target frames.")
                
                frame_idx += 1
            
            cap.release()
            if os.path.exists(video_path): 
                 os.unlink(video_path) 
            
            if video_results_list:
                st.success(f"✅ Analysis complete! Processed {len(video_results_list)} faces across {processed_frame_count} frames.")
                
                emotions = [r['emotion'] for r in video_results_list if 'emotion' in r and r['emotion'] != "Error"]
                real_faces_flags = [r['is_real'] for r in video_results_list if 'is_real' in r] 
                
                col1_res, col2_res, col3_res = st.columns(3)
                with col1_res:
                    most_common_emotion = max(set(emotions), key=emotions.count) if emotions else "N/A"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎭 Dominant Emotion</h3>
                        <h2>{most_common_emotion}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2_res:
                    authenticity_rate = (sum(real_faces_flags) / len(real_faces_flags) * 100) if real_faces_flags else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🛡️ Authenticity Rate</h3>
                        <h2>{authenticity_rate:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3_res:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📊 Total Faces Analyzed</h3>
                        <h2>{len(video_results_list)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.session_state.analysis_results.extend(video_results_list)
            else:
                st.warning("⚠️ No faces detected or processed in the video.")

def analysis_dashboard():
    st.markdown("## 📈 Analysis Dashboard")
    st.markdown("Comprehensive overview of all performed analyses and statistical insights.")
    
    if not st.session_state.analysis_results:
        st.info("📊 No analysis data available. Perform some analyses to view dashboard statistics.")
        return
    
    df = pd.DataFrame(st.session_state.analysis_results)
    
    for col in ['emotion', 'emotion_confidence', 'is_real', 'real_confidence', 'timestamp']:
        if col not in df.columns:
            if col in ['emotion_confidence', 'real_confidence']: df[col] = 0.0
            elif col == 'is_real': df[col] = False
            elif col == 'timestamp': df[col] = pd.NaT
            else: df[col] = "N/A"

    df['emotion_confidence'] = pd.to_numeric(df['emotion_confidence'], errors='coerce').fillna(0.0)
    df['real_confidence'] = pd.to_numeric(df['real_confidence'], errors='coerce').fillna(0.0)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_analyses = len(df)
        st.markdown(f"""<div class="metric-card"><h3>📊 Total Analyses</h3><h2>{total_analyses}</h2></div>""", unsafe_allow_html=True)
    
    with col2:
        valid_emotions_df = df[~df['emotion'].isin(["Error", "N/A", "Invalid ROI"])]
        unique_emotions = valid_emotions_df['emotion'].nunique() if not valid_emotions_df.empty else 0
        st.markdown(f"""<div class="metric-card"><h3>🎭 Unique Emotions</h3><h2>{unique_emotions}</h2></div>""", unsafe_allow_html=True)
    
    with col3:
        authenticity_rate = df['is_real'].mean() * 100 if 'is_real' in df.columns and not df['is_real'].empty else 0
        st.markdown(f"""<div class="metric-card"><h3>🛡️ Authenticity Rate</h3><h2>{authenticity_rate:.1f}%</h2></div>""", unsafe_allow_html=True)
    
    with col4:
        avg_confidence = valid_emotions_df['emotion_confidence'].mean() * 100 if not valid_emotions_df.empty else 0
        st.markdown(f"""<div class="metric-card"><h3>🎯 Avg Emotion Conf.</h3><h2>{avg_confidence:.1f}%</h2></div>""", unsafe_allow_html=True)
    
    if not valid_emotions_df.empty:
        st.markdown("### 📊 Emotion Distribution")
        emotion_counts = valid_emotions_df['emotion'].value_counts()
        if not emotion_counts.empty:
            fig_pie = px.pie(emotion_counts, values=emotion_counts.values, names=emotion_counts.index, 
                         title="Distribution of Detected Emotions")
            st.plotly_chart(fig_pie, use_container_width=True)
    
    df_timeline = df.dropna(subset=['timestamp'])
    if not df_timeline.empty:
        st.markdown("### 📈 Analysis Timeline")
        df_timeline['hour'] = df_timeline['timestamp'].dt.hour
        hourly_counts = df_timeline['hour'].value_counts().sort_index()
        if not hourly_counts.empty:
            fig_line = px.line(hourly_counts, x=hourly_counts.index, y=hourly_counts.values,
                             title="Analysis Activity by Hour",
                             labels={'index': 'Hour of Day', 'value': 'Number of Analyses'}) # Updated labels for direct Series plotting
            st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown("### 📋 Recent Analysis Results (Last 10)")
    recent_results_df = df.tail(10) 
    if not recent_results_df.empty:
        display_cols_options = ['id', 'emotion', 'emotion_confidence', 'is_real', 'real_confidence', 'display_spoof_confidence', 'source', 'timestamp', 'frame']
        cols_to_show_in_df = [col for col in display_cols_options if col in recent_results_df.columns]
        
        formatted_df = recent_results_df[cols_to_show_in_df].copy()
        for col_name in ['emotion_confidence', 'real_confidence', 'display_spoof_confidence']:
            if col_name in formatted_df.columns:
                formatted_df[col_name] = pd.to_numeric(formatted_df[col_name], errors='coerce')
                formatted_df[col_name] = (formatted_df[col_name] * 100).map('{:.1f}%'.format)
        st.dataframe(formatted_df, use_container_width=True)
    
    if st.button("🗑️ Clear Analysis History", key="clear_history", use_container_width=True):
        st.session_state.analysis_results = []
        st.rerun()

if __name__ == "__main__":
    main()
