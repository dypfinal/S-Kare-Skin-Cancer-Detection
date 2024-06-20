import streamlit as st
import PIL
import random
import json
import torch
import spacy
import re
import numpy as np
import nltk

nltk.download("punkt")
from nltk.stem.porter import PorterStemmer
import pickle
import tensorflow as tf
import numpy as np
import os
from glob import glob
from PIL import Image
import pandas as pd
import time
import speech_recognition as sr


# Load the scaler object from the pickle file
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
# Load the encoder object for localization from the pickle file
with open("encoder_localization.pkl", "rb") as f:
    localization_encoder = pickle.load(f)
# Load the encoder object for sex from the pickle file
with open("encoder_sex.pkl", "rb") as f:
    sex_encoder = pickle.load(f)
mixed_data_model = tf.keras.models.load_model(
    "alexnet_augmented_200_100_0.5_thrice_0.8264_0.8309_0.9573.h5"
)

label_dict = {0: "Melanocytic nevi", 1: "Melanoma", 2: "No_Skin_Disease", 3: "Others"}
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag


body_parts = [
    "head",
    "neck",
    "shoulder",
    "arm",
    "elbow",
    "wrist",
    "hand",
    "finger",
    "chest",
    "stomach",
    "abdomen",
    "back",
    "waist",
    "hip",
    "leg",
    "knee",
    "ankle",
    "foot",
    "toe",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intents.json", "r") as json_data:
    intents = json.load(json_data)

FILE = "chatbot_model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]


class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


def extract_age(text):
    patterns = [
        r"\b(\d{1,3})\s*(years? old|years|yrs|yo)\b",
        r"\b(age|aged)\s*(\d{1,3})\b",
        r"\bi\s*am\s*(\d{1,3})\b",
        r"\b(\d{1,3})\s*years?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


st.header("🩺 Medbot: Your Personal Skin Care Assistant")
st.subheader("Please Upload Relevant Information Below for a diagnosis.")

uploaded_file = st.file_uploader("Upload a file")

# Create columns to arrange elements side by side
col1, col2, col3 = st.columns(3)

with col1:
    slider_value = st.slider(
        "Please select your Age:", min_value=0, max_value=100, value=50
    )
with col2:
    selectbox_value1 = st.selectbox("Gender", ["male", "female"])
with col3:
    selectbox_value2 = st.selectbox(
        "Body Part/Region: ",
        [
            "hand",
            "lower extremity",
            "back",
            "face",
            "trunk",
            "genital",
            "neck",
            "ear",
            "unknown",
            "acral",
            "scalp",
            "foot",
            "chest",
            "abdomen",
            "upper extremity",
        ],
    )
######################################### -                ####################
# predict = st.button("Predict")


def open_support_ticket():
    email_link = "http://127.0.0.1:8000/"
    webbrowser.open(email_link)


# Create columns to place buttons side by side
col1, col2 = st.columns(2)

with col1:
    predict = st.button("Predict")

with col2:
    contact_us = st.button("Book an Appointment!", on_click=open_support_ticket)
if predict:

    # Create a dictionary with the data
    data = {
        "age": [slider_value, slider_value],
        "sex": [selectbox_value1, selectbox_value1],
        "localization": [selectbox_value2, selectbox_value2],
    }

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Preprocess the data
    df["age"] = scaler.transform(df[["age"]])
    df["sex"] = sex_encoder.transform(df[["sex"]])
    df["localization"] = localization_encoder.transform(df[["localization"]])

    image_bytes = uploaded_file.read()
    # Open the image using PIL
    import io

    img = np.asarray(Image.open(io.BytesIO(image_bytes)).resize((200, 100)))
    img_arr = [img, img]

    response_predict = ""
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)
            if i == 25:
                status_text.text("Preprocessing data...")
            elif i == 50:
                status_text.text("Running predictions...")
            elif i == 75:
                status_text.text("Finalizing results...")
        y_pred = mixed_data_model.predict([df, np.array(img_arr)])
        max_prob_indices = np.argmax(y_pred, axis=1)

        # Initialize mapped predictions list
        mapped_predictions = []

        # Map the values to their corresponding labels
        for pred in max_prob_indices:
            if (y_pred[0][pred] > 0.75) and (
                label_dict[pred] == "Melonoma" or label_dict[pred] == "Others"
            ):
                mapped_predictions.append(label_dict[pred])
            else:
                mapped_predictions.append(None)
        print("Mapped predictions:", mapped_predictions)
        if mapped_predictions[0]:
            response_predict += f"\nHigh Chances for Skin Cancer. Please schedule a visit to the nearest doctor."
            st.markdown(
                f'<div style="background-color: #fff3cd; padding: 10px; border-radius: 5px;">'
                f'<span style="color: #856404;">{response_predict} Please visit '
                f"Neares S-Kare Kiosk for booking an appointment or call on +91xxxxxxxx</span></div>",
                unsafe_allow_html=True,
            )
        else:
            response_predict += "Skin is healthy! Keep it Up!"
            st.markdown(
                f'<div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">'
                f'<span style="color: #155724;">Skin is healthy! Keep it Up! ',
                unsafe_allow_html=True,
            )
    except:
        response_predict += "\nERROR: Please check input"
        st.error("Please Check Input!")


###################################                     #######################

if "messages" not in st.session_state:
    st.session_state.messages = []
st.subheader("💊MedBot: The Interactive Skin Assistant")

if st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            st.success("Voice Input: " + query)
            return query
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            st.error(
                f"Could not request results from Google Speech Recognition service; {e}"
            )


if prompt := st.chat_input("Please Describe your Problems."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})


voice_ip = st.button("Voice Input", key="voice_input", help="Click to input via voice")
# voice_ip._st_button.markdown('<style>div.row-widget.stButton > button { background-color: #f0f0f0; color: #000; }</style>', unsafe_allow_html=True)
if not prompt:
    if voice_ip:
        prompt = get_voice_input()
    else:

        prompt = "Hello Test Input"
        prompt = None
print(prompt)
bot_name = "MedBot"

if prompt:
    sentence_tokenized = tokenize(prompt)
    X = bag_of_words(sentence_tokenized, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent["responses"])
    else:
        response = "I do not understand..."

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

import webbrowser
