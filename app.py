import streamlit as st

st.title('Customer Sentiment Analysis App')
st.write('Welcome to the multimodal customer sentiment analysis application.')

print('app.py created successfully with basic Streamlit structure.')

# Import all necessary libraries for the AI model functions
import pandas as pd
import torch
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import speech_recognition as sr
import librosa
import cv2
from fer.fer import FER
import numpy as np
import io
import tempfile
import os
from gtts import gTTS
from pydub import AudioSegment

# 2. Define the text_sentiment_pipeline
@st.cache_resource
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

text_sentiment_pipeline = get_sentiment_pipeline()

# 3. Define analyze_text_sentiment function
def analyze_text_sentiment(text):
    if pd.isna(text):
        return "No review", 0.0
    result = text_sentiment_pipeline(text)[0]
    return result['label'], result['score']

# 4. Define analyze_voice_sentiment function
def analyze_voice_sentiment(audio_input):
    # Handle BytesIO object by saving to a temporary file
    temp_audio_path = None
    if isinstance(audio_input, io.BytesIO):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_input.read())
            temp_audio_path = tmp_file.name
        audio_file_to_process = temp_audio_path
    else:
        audio_file_to_process = audio_input

    transcribed_text = "Could not understand audio"
    sentiment_label = "neutral"
    sentiment_score = 0.0
    pitch = 0.0
    energy = 0.0

    try:
        y, sr_rate = librosa.load(audio_file_to_process, sr=None)
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file_to_process) as source:
            audio_data = recognizer.record(source)
            try:
                transcribed_text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                transcribed_text = "Could not understand audio"

        sentiment_label, sentiment_score = analyze_text_sentiment(transcribed_text)
        
        # Pitch detection can be unreliable for short or silent audio, handle potential errors
        try:
            pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr_rate)
            pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0
        except Exception:
            pitch = 0.0

        energy = np.mean(librosa.feature.rms(y=y)) if len(y) > 0 else 0.0

    except Exception as e:
        st.warning(f"Error processing audio: {e}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return {
        'transcribed_text': transcribed_text,
        'sentiment': sentiment_label,
        'confidence': sentiment_score,
        'pitch_hz': pitch,
        'energy': energy
    }

# 5. Define analyze_video_sentiment function
def analyze_video_sentiment(video_file):
    detector = FER(mtcnn=True)  # Advanced face detection
    cap = cv2.VideoCapture(video_file)
    emotions = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:  # Analyze every 30th frame for efficiency
            result = detector.detect_emotions(frame)
            if result:
                # Ensure result[0]['emotions'] is not empty
                if result[0]['emotions']:
                    top_emotion = max(result[0]['emotions'], key=result[0]['emotions'].get)
                    emotions.append(top_emotion)
    cap.release()
    if emotions:
        dominant_emotion = max(set(emotions), key=emotions.count)
        emotion_counts = {emo: emotions.count(emo) for emo in set(emotions)}
        return dominant_emotion, emotion_counts
    return "No emotions detected", {}

# 6. Initialize GPT-2 models
@st.cache_resource
def get_gpt2_models():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

generator_tokenizer, generator_model = get_gpt2_models()

# 7. Define generate_insights function
def generate_insights(prompt, max_length=150):
    inputs = generator_tokenizer.encode(prompt, return_tensors="pt")
    outputs = generator_model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95)
    return generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

st.write('AI model functions and definitions added to app.py.')

st.header('1. Customer Data Upload')
uploaded_file = st.file_uploader("Upload your Customer_Sentiment.csv file", type=["csv"])

st.header('2. Text Sentiment Analysis')
text_input = st.text_area("Enter customer review text for sentiment analysis:", "This product is fantastic!")

st.header('3. Voice Sentiment Analysis')
voice_file = st.file_uploader("Upload an audio file (WAV) for voice sentiment analysis:", type=["wav"])

st.header('4. Video Sentiment Analysis')
video_file = st.file_uploader("Upload a video file (MP4) for video sentiment analysis:", type=["mp4"])

st.write('Streamlit input widgets added to app.py.')

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.header('5. Analysis Results')

if uploaded_file is not None:
    st.subheader('5.1. CSV Data Analysis')
    try:
        df_uploaded = pd.read_csv(uploaded_file)

        # Create a sample of the DataFrame for faster processing
        sample_size = min(5000, len(df_uploaded))
        df_sample = df_uploaded.sample(n=sample_size, random_state=42).copy() # Use .copy() to avoid SettingWithCopyWarning

        st.write(f"Analyzing a sample of {sample_size} reviews from the uploaded CSV.")

        # Convert review_text from the sample to a list for batch processing
        review_texts_sample = df_sample['review_text'].tolist()

        # Perform sentiment analysis on all sampled texts in one go
        batch_results_sample = text_sentiment_pipeline(review_texts_sample)

        # Extract predicted labels and scores for the sample
        predicted_labels_sample = [result['label'] for result in batch_results_sample]
        predicted_scores_sample = [result['score'] for result in batch_results_sample]

        # Assign to DataFrame columns in the sample
        df_sample['predicted_sentiment'] = predicted_labels_sample
        df_sample['sentiment_score'] = predicted_scores_sample

        # Compare with existing sentiment in the sample
        df_sample['sentiment_match'] = df_sample['sentiment'] == df_sample['predicted_sentiment'].str.lower()

        st.write("Sampled Data with Sentiment Analysis Results:")
        st.dataframe(df_sample[['review_text', 'sentiment', 'predicted_sentiment', 'sentiment_score', 'sentiment_match']].head(10))

        # Generate and display visualizations
        st.subheader('5.1.1. Visualizations from CSV Data')

        # Graph 1: Sentiment Distribution Pie Chart
        fig1 = px.pie(df_sample, names='predicted_sentiment', title='Overall Sentiment Distribution', hole=0.3)
        st.plotly_chart(fig1)

        # Graph 2: Sentiment by Product Category (Bar Chart)
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df_sample, x='product_category', hue='predicted_sentiment', palette='viridis')
        plt.title('Sentiment by Product Category')
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Graph 3: Average Rating by Region (Box Plot)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_sample, x='region', y='customer_rating', hue='region', palette='Set2', legend=False)
        plt.title('Average Customer Rating by Region')
        st.pyplot(plt)

        # Graph 4: Response Time vs. Issue Resolved (Scatter Plot)
        fig4 = px.scatter(df_sample, x='response_time_hours', y='customer_rating', color='issue_resolved',
                          title='Response Time vs. Rating (Colored by Issue Resolved)',
                          hover_data=['product_category', 'sentiment'])
        st.plotly_chart(fig4)

        # Graph 5: Complaint Registered by Age Group (Stacked Bar)
        age_complaints = df_sample.groupby(['age_group', 'complaint_registered']).size().unstack()
        plt.figure(figsize=(10, 6))
        age_complaints.plot(kind='bar', stacked=True)
        plt.title('Complaints by Age Group')
        plt.ylabel('Count')
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error processing CSV file: {e}")

if text_input:
    st.subheader('5.2. Text Sentiment Analysis Results')
    text_sentiment_label, text_sentiment_score = analyze_text_sentiment(text_input)
    st.write(f"**Input Text:** {text_input}")
    st.write(f"**Predicted Sentiment:** {text_sentiment_label}")
    st.write(f"**Confidence Score:** {text_sentiment_score:.4f}")

if voice_file is not None:
    st.subheader('5.3. Voice Sentiment Analysis Results')
    try:
        audio_bytes = voice_file.read()
        voice_results = analyze_voice_sentiment(io.BytesIO(audio_bytes))
        st.write(f"**Transcribed Text:** {voice_results['transcribed_text']}")
        st.write(f"**Predicted Sentiment:** {voice_results['sentiment']}")
        st.write(f"**Confidence Score:** {voice_results['confidence']:.4f}")
        st.write(f"**Average Pitch (Hz):** {voice_results['pitch_hz']:.2f}")
        st.write(f"**Average Energy:** {voice_results['energy']:.4f}")
    except Exception as e:
        st.error(f"Error processing voice file: {e}")

import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns

if video_file is not None:
    st.subheader('5.4. Video Sentiment Analysis Results')
    try:
        # Save the uploaded video file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            temp_video_path = tmp_file.name
        
        st.write(f"Analyzing video file: {video_file.name}")
        dominant, counts = analyze_video_sentiment(temp_video_path)
        
        st.write(f"**Dominant Emotion:** {dominant}")
        st.write("**Emotion Counts:**", counts)
        
        # Graph for video emotions
        if counts:
            plt.figure(figsize=(8, 4))
            sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette='coolwarm')
            plt.title('Emotion Distribution in Video')
            plt.xlabel('Emotions')
            plt.ylabel('Frame Count')
            st.pyplot(plt)
        else:
            st.write("No emotions detected in the video.")

    except Exception as e:
        st.error(f"Error processing video file: {e}")
    finally:
        # Clean up the temporary file
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.remove(temp_video_path)

if 'df_sample' in locals(): # Only run if CSV data was processed and df_sample exists
    st.subheader('5.5. Generative AI Insights')

    # Example: Generate insights from CSV sentiments
    negative_reviews_df = df_sample[df_sample['predicted_sentiment'] == 'NEGATIVE']['review_text']
    if not negative_reviews_df.empty:
        # Ensure we don't try to sample from an empty Series
        sample_size_neg = min(5, len(negative_reviews_df))
        negative_reviews = negative_reviews_df.sample(sample_size_neg, random_state=42).tolist()
        if negative_reviews:
            prompt = f"Based on these negative customer reviews: {', '.join(negative_reviews)}\nSuggest improvements for the brand:"
            insights = generate_insights(prompt)
            st.write("**Insights from Negative Reviews:**")
            st.write(insights)
        else:
            st.write("No negative reviews found in the sample to generate insights.")
    else:
        st.write("No negative reviews found in the sample to generate insights.")

    # Additional: Overall summary
    positive_count = len(df_sample[df_sample['predicted_sentiment']=='POSITIVE'])
    negative_count = len(df_sample[df_sample['predicted_sentiment']=='NEGATIVE'])
    neutral_count = len(df_sample[df_sample['predicted_sentiment']=='NEUTRAL'])

    overall_prompt = f"Summary of sentiments: Positive: {positive_count}, Negative: {negative_count}, Neutral: {neutral_count}.\nKey recommendations:"
    summary_insights = generate_insights(overall_prompt)
    st.write("**Overall Summary Insights:**")
    st.write(summary_insights)


st.write('Streamlit output widgets for Generative AI insights added to app.py.')
