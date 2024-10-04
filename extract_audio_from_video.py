import cv2
import torch
from torchvision import transforms
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import speech_recognition as sr
from moviepy.editor import *
import os

# Initialize models and transformations
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_frames(video_path, frame_rate=1):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frames = []
    count = 0
    while success:
        if count % frame_rate == 0:
            frames.append(image)
        success, image = vidcap.read()
        count += 1
    return frames

def generate_caption(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def extract_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_file = "temp_audio.wav"
    audio.write_audiofile(audio_file, codec="pcm_s16le")

    # Use speech recognition to transcribe the audio
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)
        transcription = r.recognize_google(audio_data)
    return transcription

def multimodal_text_output(video_path):
    # Extract video frames and generate captions
    frames = extract_frames(video_path)
    captions = [generate_caption(frame) for frame in frames]
    video_narrative = " ".join(captions)

    # Extract and transcribe audio from the video
    audio_transcription = extract_audio_from_video(video_path)

    # Combine visual and audio information
    final_output = f"Video Narrative: {video_narrative}\nAudio Transcription: {audio_transcription}"
    return final_output

# Usage
video_path = "C:/Users/zconj/Documents/ue5.mp4"
output_text = multimodal_text_output(video_path)
print(output_text)
