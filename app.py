import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from deepface import DeepFace
import cv2
import numpy as np

st.set_page_config(page_title="Live Emotion Detector AI", page_icon="🎭", layout="centered")

st.title("🎭 Live Real-Time Emotion Detector AI")
st.write("Allow camera access below to detect your emotions live!")

# Mapping emotions to emojis
emoji_map = {
    "happy": "😊", "sad": "😢", "angry": "😠", 
    "surprise": "😲", "neutral": "😐", "fear": "😨", "disgust": "🤢"
}

class EmotionTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the incoming video frame into an OpenCV BGR image array
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # DeepFace analyzes the current frame directly from the video stream
            result = DeepFace.analyze(
                img_path=img,
                actions=['emotion'],
                enforce_detection=False
            )
            
            # Extract dominant emotion
            dominant_emotion = result['dominant_emotion']
            emoji = emoji_map.get(dominant_emotion, "🙂")
            text = f"{dominant_emotion.capitalize()} {emoji}"
            
            # Draw the detected emotion text straight onto the video frame screen
            cv2.putText(img, text, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            
        except Exception as e:
            # If a frame fails to process, pass it through seamlessly without breaking the stream
            pass
            
        return img

# Spin up the live WebRTC video streamer widget interface
webrtc_streamer(key="emotion-detection-stream", video_transformer_factory=EmotionTransformer)
