import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from deepface import DeepFace
import cv2
import av

st.set_page_config(page_title="Live Emotion Detector AI", page_icon="🎭", layout="centered")

st.title("🎭 Live Real-Time Emotion Detector AI")
st.write("Keep your face centered and clearly lit in front of the lens!")

# Global dictionary mapping labels to visual emojis
emoji_map = {
    "happy": "😊", "sad": "😢", "angry": "😠", 
    "surprise": "😲", "neutral": "😐", "fear": "😨", "disgust": "🤢"
}

class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert incoming frame to standard OpenCV BGR matrix
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Run DeepFace analysis directly on the frame image matrix
            results = DeepFace.analyze(
                img_path=img,
                actions=['emotion'],
                enforce_detection=True  # Find tracking anchors first
            )
            
            # CRITICAL FIX: DeepFace outputs an evaluation array list format
            if isinstance(results, list) and len(results) > 0:
                analysis = results[0]  # Read the primary target dictionary out of the list container
                dominant_emotion = analysis['dominant_emotion']
                emoji = emoji_map.get(dominant_emotion, "🙂")
                display_text = f"{dominant_emotion.capitalize()} {emoji}"
                
                # Extract face coordinates to draw tracking box layout
                region = analysis['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
                # Draw the tracking bounding box and label onto screen matrix
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                
        except Exception as e:
            # Display background diagnostic loop info rather than crashing out completely
            cv2.putText(img, "Looking for face...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Initialize WebRTC Stream Component with secure global STUN configurations
webrtc_streamer(
    key="live-emotion-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EmotionProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:://google.com"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
