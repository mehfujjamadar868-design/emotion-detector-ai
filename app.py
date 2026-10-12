import streamlit as st
from deepface import DeepFace
import numpy as np
import cv2


st.set_page_config(page_title="Emotion Detector AI", page_icon="🎭", layout="centered")

st.title("🎭 Emotion Detector AI")
st.write("Upload an image below to let the AI analyze and detect the emotional state.")


uploaded_file = st.file_uploader("Choose an image profile...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    
    st.image(opencv_image, channels="BGR", caption="Uploaded Image", use_container_width=True)
    
   
    if st.button("Analyze Emotion 🚀"):
        with st.spinner("Analyzing facial expressions..."):
            try:
               
                result = DeepFace.analyze(
                    img_path=opencv_image,
                    actions=['emotion'],
                    enforce_detection=False
                )
                
                dominant_emotion = result[0]['dominant_emotion']
                
               
                emoji_map = {
                    "happy": "😊", "sad": "😢", "angry": "😠", 
                    "surprise": "😲", "neutral": "😐", "fear": "😨", "disgust": "🤢"
                }
                emoji = emoji_map.get(dominant_emotion, "🙂")
                
                st.success(f"**Detected Emotion:** {dominant_emotion.capitalize()} {emoji}")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
