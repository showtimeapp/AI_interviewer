# import cv2
# import mediapipe as mp
# from deepface import DeepFace
# import pandas as pd
# import time
# import streamlit as st
# import tempfile

# st.set_page_config(layout="wide")
# # Initialize MediaPipe FaceMesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils

# # Engagement emotions
# FOCUSED_EMOTIONS = ["happy", "neutral", "surprise"]
# data_log = []  # To store data for CSV export

# # Function to calculate gaze direction and head tilt
# def calculate_focus(face_landmarks, frame_width, frame_height):
#     left_eye_x = face_landmarks.landmark[33].x * frame_width  # Left eye corner
#     right_eye_x = face_landmarks.landmark[263].x * frame_width  # Right eye corner
#     nose_x = face_landmarks.landmark[1].x * frame_width  # Nose tip
#     nose_y = face_landmarks.landmark[1].y * frame_height  # Nose vertical position
#     chin_y = face_landmarks.landmark[152].y * frame_height  # Chin position
    
#     head_tilt = abs(nose_y - chin_y)
    
#     if left_eye_x < nose_x < right_eye_x:
#         gaze_score = 100  # Looking at the screen
#     elif nose_x < left_eye_x:
#         gaze_score = 60  # Looking left
#     else:
#         gaze_score = 60  # Looking right
    
#     # Adjust score based on head tilt
#     if head_tilt > frame_height * 0.05:

#         gaze_score -= 20  # Reduce focus if head is tilted too much
    
#     return max(0, gaze_score)

# def analyze_focus(frame):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)
#     num_faces = 0
    
#     if results.multi_face_landmarks:
#         num_faces = len(results.multi_face_landmarks)
#         for face_landmarks in results.multi_face_landmarks:
#             # Draw face landmarks
#             mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
#                                       landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
            
#             # Get focus score based on gaze and head tilt
#             focus_score = calculate_focus(face_landmarks, frame.shape[1], frame.shape[0])
            
#             # Analyze face emotion
#             try:
#                 analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
#                 emotion = analysis[0]['dominant_emotion']
#             except:
#                 emotion = "unknown"
            
#             # Adjust focus score based on emotion
#             if emotion not in FOCUSED_EMOTIONS:
#                 focus_score -= 20  # Reduce score if emotion is disengaged
#             focus_score = max(0, focus_score)  # Ensure focus score is not negative
            
#             # Display results on frame
#             text = f"Emotion: {emotion} | Focus Score: {focus_score}"
#             cv2.putText(frame, text, (30, 30 * (num_faces + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
#             # Log data
#             data_log.append([emotion, focus_score, num_faces])

#     return frame, num_faces, focus_score, emotion

# # # Start webcam
# # cap = cv2.VideoCapture(0)
# # start_time = time.time()

# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
    
# #     frame, num_faces = analyze_focus(frame)
# #     cv2.imshow("Focus Detection", frame)
    
# #     # Stop after a certain time (optional)
# #     if time.time() - start_time > 60:  # Run for 60 seconds
# #         break
    
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()

# # # Save results to CSV
# # df = pd.DataFrame(data_log, columns=["Emotion", "Focus Score", "No. of Faces"])
# # df.to_csv("focus_analysis.csv", index=False)
# # print("CSV file saved: focus_analysis.csv")
# def main():
#     # st.title("🔍 Real-Time Focus Detection")
#     # st.markdown("Detects engagement based on gaze & emotion.")
    
#     # if st.button("Start Webcam"):
#         cap = cv2.VideoCapture(0)
#         start_time = time.time()
#         stframe = st.empty()
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame, num_faces, focus_score, emotion = analyze_focus(frame)
            
#             # Display in Streamlit
#             stframe.image(frame, channels="BGR")
#             # st.write(f"**Emotion:** {emotion} | **Focus Score:** {focus_score} | **No. of Faces:** {num_faces}")
            

#             if time.time() - start_time > 120:
#                 break

#         cap.release()
#         df = pd.DataFrame(data_log, columns=["Emotion", "Focus Score", "No. of Faces"])
#         csv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
#         df.to_csv(csv_file, index=False)
#         st.success("Session complete! Download your focus analysis data below.")
#         st.download_button("Download CSV", csv_file, "focus_analysis.csv")
        
import cv2
import mediapipe as mp
from deepface import DeepFace
import pandas as pd
import time
import streamlit as st
import json
import tempfile
import os
from pathlib import Path
from groq import Groq
from faster_whisper import WhisperModel
from gtts import gTTS
import torch

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Engagement emotions
FOCUSED_EMOTIONS = ["happy", "neutral", "surprise"]
data_log = []  # To store data for CSV export

# Function to calculate gaze direction and head tilt
def calculate_focus(face_landmarks, frame_width, frame_height):
    left_eye_x = face_landmarks.landmark[33].x * frame_width  # Left eye corner
    right_eye_x = face_landmarks.landmark[263].x * frame_width  # Right eye corner
    nose_x = face_landmarks.landmark[1].x * frame_width  # Nose tip
    nose_y = face_landmarks.landmark[1].y * frame_height  # Nose vertical position
    chin_y = face_landmarks.landmark[152].y * frame_height  # Chin position
    
    head_tilt = abs(nose_y - chin_y)
    
    if left_eye_x < nose_x < right_eye_x:
        gaze_score = 100  # Looking at the screen
    elif nose_x < left_eye_x:
        gaze_score = 60  # Looking left
    else:
        gaze_score = 60  # Looking right
    
    # Adjust score based on head tilt
    if head_tilt > frame_height * 0.05:
        gaze_score -= 20  # Reduce focus if head is tilted too much
    
    return max(0, gaze_score)

def analyze_focus(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    num_faces = 0
    focus_score = 0
    emotion = "no_face"
    
    if results.multi_face_landmarks:
        num_faces = len(results.multi_face_landmarks)
        for face_landmarks in results.multi_face_landmarks:
            # Draw face landmarks
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
            
            # Get focus score based on gaze and head tilt
            focus_score = calculate_focus(face_landmarks, frame.shape[1], frame.shape[0])
            
            # Analyze face emotion
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = analysis[0]['dominant_emotion']
            except Exception as e:
                st.warning(f"Error in emotion detection: {e}")
                emotion = "unknown"
            
            # Adjust focus score based on emotion
            if emotion not in FOCUSED_EMOTIONS:
                focus_score -= 20  # Reduce score if emotion is disengaged
            focus_score = max(0, focus_score)  # Ensure focus score is not negative
            
            # Display results on frame
            text = f"Emotion: {emotion} | Focus Score: {focus_score}"
            cv2.putText(frame, text, (30, 30 * (num_faces + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else: 
        # No face detected
        st.warning("No face detected. Please ensure you are in front of the camera.")
        focus_score = 0
        emotion = "no_face"
    
    # Log data
    data_log.append([emotion, focus_score, num_faces])
    
    return frame, num_faces, focus_score, emotion

def main():
    # st.title("Focus Analysis")
    
    # if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            frame, num_faces, focus_score, emotion = analyze_focus(frame)
            
            # Display in Streamlit
            stframe.image(frame, channels="BGR")

            if time.time() - start_time > 900:
                break

        cap.release()

        # Convert data log to DataFrame
        df = pd.DataFrame(data_log, columns=["Emotion", "Focus Score", "No. of Faces"])

        # Save DataFrame to CSV
        try:
            csv_file = os.path.join(tempfile.gettempdir(), "focus_analysis.csv")
            df.to_csv(csv_file, index=False)

            st.success("Session complete! Download your focus analysis data below.")
            with open(csv_file, "rb") as file:
                st.download_button(
                    label="Download CSV", 
                    data=file, 
                    file_name="focus_analysis.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error saving CSV: {e}")

# Config - Set Your Groq API Key Here
GROQ_API_KEY = "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"

# Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)

if "domain" not in st.session_state:
    st.session_state["domain"] = None

# Personality / System Message
def generate_personality(domain):
    return f"""
    Task: You Are RexAI, Assume the role of a Senior {domain} at Showtime. You were assigned to follow these rules:

    Rules to Follow:
    1. Maintain a strict focus on the {domain} interview; do not deviate from the topic.
    2. If the user goes off-topic, gently remind them to stay focused on the interview.
    3. Create and present up-to-date questions relevant to the field of {domain}.

    Interview Structure:
    The interview will consist of three rounds:
    - Round 1: This round will include 10 challenging multiple-choice questions. You are to ask these questions to the user.
    - Round 2: This round will consist of 4 short-answer questions. You will ask these questions to the user.
    - Round 3: This round will involve 2 detailed, long-form questions. You will ask these in-depth questions to the user.

    Evaluation Criteria:
    - Round 1: Each question is worth 1 point. Award 1 point for each correct answer and 0 points for incorrect answers. If the user scores 5 points or fewer, end the interview with the message, "Sorry, we won't be moving forward with you."
    - Round 2: Each question is worth 2 points. Award 2 points for each correct answer and 0 points for incorrect answers. If the user scores 4 points or fewer, end the interview with the message, "Sorry, we won't be moving forward with you."
    - Round 3: Each question is worth 3 points. Award 3 points for each correct answer and 0 points for incorrect answers. If the user scores 3 points or fewer, end the interview with the message, "Sorry, we won't be moving forward with you."

    - After every Question answer tell him how many total points he earned and how any total question left in that Round and how many more Points he need to qualify that round
    - If the user answer incorrectly just give a samll explaination 1-2 sentence why it is wrong and which is correct and why
    - If the user fails in any round, terminate the session and start again from Round 1.
    - If the user successfully passes Round 1, proceed to Round 2. However, if the user fails in Round 2, terminate the session and restart from Round 1.
    - If the user passes both Round 1 and Round 2 but fails in Round 3, terminate the session and begin again from Round 1. 

    Please adhere strictly to these guidelines!
    """

# Ask for domain selection if not chosen yet
st.title("🧑‍💻 Interview Bot - RexAI")

if st.session_state["domain"] is None:
    # st.title("🧑‍💻 Interview Bot - RexAI")
    domain = st.selectbox("Select Domain", ["Political Campaign Team","Political Research Team","Data Science", "Machine Learning", "Software Engineering", "Cybersecurity", "Cloud Computing","Web Developer","Data Analyst"])
    if st.button("Start Interview"):
        st.session_state["domain"] = domain
        st.session_state["messages"] = [{"role": "system", "content": generate_personality(domain)}]
        st.rerun()
# Helper Functions
def save_history_to_json(history, file_path="history.json"):
    with open(file_path, "w") as file:
        json.dump(history, file, indent=4)

def load_history_from_json(file_path="history.json"):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return []

def groq_chat_completion(messages):
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages
    )
    return response.choices[0].message.content

def transcribe_audio(audio_file_path):
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_file_path)
    return " ".join(segment.text for segment in segments)

def generate_audio(text):
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        tts.save(temp_file.name)
        return temp_file.name
    
# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages =[{"role": "system", "content": generate_personality(domain)}]

# Inject Custom HTML & CSS for Styling
st.markdown("""
    <style>
    .chat-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        max-height: 400px;
        overflow-y: auto;
        margin-bottom: 10px;
    }
    .chat-bubble {
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 20px;
        max-width: 80%;
        font-size: 14px;
        line-height: 1.5;
        word-wrap: break-word;
    }
    .user-bubble {
        background-color: #d1e7dd;
        align-self: flex-end;
    }
    .bot-bubble {
        background-color: #f8d7da;
        align-self: flex-start;
    }
    .input-box {
        width: 100%;
        border: 1px solid #ccc;
        padding: 10px;
        border-radius: 5px;
        font-size: 14px;
    }
    .send-btn {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 14px;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Choose Mode
chat_mode = st.radio("Choose Mode:", ["Text Chat", "Speech Chat"])

# Chat Container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display Chat History (HTML Bubble Style)
for message in st.session_state.messages:
    if message["role"] == "system":
        continue

    bubble_class = "user-bubble" if message["role"] == "user" else "bot-bubble"
    st.markdown(f'<div class="chat-bubble {bubble_class}">{message["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Text Chat Mode
if chat_mode == "Text Chat":

    # Use session state to store and manage input
    user_input = st.text_input("Your message:", key="chat_input")

    # Slightly adjust button placement
    st.write("") # Add a blank line to align with input
    send_clicked = st.button("Send")
    
    # Process message when send is clicked
    if send_clicked and user_input:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get AI response
        response_text = groq_chat_completion(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # Save history
        save_history_to_json(st.session_state.messages)
        
        # Clear the input (this approach avoids modifying session state directly)
        st.experimental_set_query_params(chat_input="")
        
        # Force refresh
        st.rerun()

# Store text input in session state
    # st.text_input("Your message:", key="user_input")

# Speech Chat Mode
elif chat_mode == "Speech Chat":
    audio_file = st.file_uploader("Upload your voice message (MP3 only)", type=["mp3"])

    if st.button("Process Audio"):
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                temp_audio_file.write(audio_file.read())
                temp_audio_file_path = temp_audio_file.name

            transcript = transcribe_audio(temp_audio_file_path)
            st.session_state.messages.append({"role": "user", "content": transcript})

            response_text = groq_chat_completion(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

            audio_response_path = generate_audio(response_text)

            save_history_to_json(st.session_state.messages)

            st.write("### Transcript")
            st.write(transcript)

            st.write("### RexAI Response")
            st.write(response_text)

            st.audio(audio_response_path, format="audio/mp3")

            # st.experimental_rerun()    
if __name__ == "__main__":
    main()
