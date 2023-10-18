import os
import torch
import torchaudio
from pydub import AudioSegment
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import whisper
from torchaudio.transforms import Resample
from whisper.audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
import subprocess
import os
import cv2
import shutil
from tqdm import tqdm
import easyocr
import Levenshtein
import unicodedata
import re
import pandas as pd
import json

#video_file = "horror/from_meir.mp4"
video_file = r"horror/from_meir.mp4"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")
whisper_model_size = "large-v2"
# whisper_model_size = "base"
TARGET_SR = 16000

def extract_audio_from_video(video_path, audio_output_path):

    if os.path.exists(audio_output_path):
        os.remove(audio_output_path)

    command = [
        'ffmpeg',
        '-i', video_path,
        '-q:a', '0',
        '-map', 'a',
        audio_output_path
    ]
    subprocess.run(command)

def get_stt(audio_file, res_dict):

    whisper_model   = whisper.load_model(whisper_model_size)

    start_second  = 0
    end_second    = 29
    full_duration = AudioSegment.from_mp3(audio_file).duration_seconds
    counter       = 1
    while True:
        current_audio = AudioSegment.from_mp3(audio_file)[start_second * 1000: end_second * 1000]
        current_audio = current_audio.set_frame_rate(16000)
        current_audio.export(out_f="tmp_whisper.wav", format="wav")

        audio           = whisper.load_audio("tmp_whisper.wav")
        audio           = whisper.pad_or_trim(audio)
        mel             = whisper.log_mel_spectrogram(audio).to(DEVICE)

        _, probs    = whisper_model.detect_language(mel)
        detect_lang = max(probs, key=probs.get)
        detect_prob = round(probs[detect_lang], 2)

        he_prob = probs['he']
        ar_prob = probs['ar']

        if he_prob > ar_prob:
            language = 'he'
        else:
            language = 'ar'

        whisper_options = whisper.DecodingOptions(fp16=True, task='transcribe', beam_size=5, language=language)
        result          = whisper.decode(whisper_model, mel, whisper_options)
        src_text        = result.text
    
        whisper_options = whisper.DecodingOptions(fp16=True, task='translate', beam_size=5, language=language)
        result          = whisper.decode(whisper_model, mel, whisper_options)
        trs_text        = result.text

        # Print the results for now
        print(f"{[{counter}]} Detected Language: {detect_lang}")
        print(f"{[{counter}]} Detected Language Probability: {detect_prob}")
        print(f"{[{counter}]} Transcription:    {src_text}")
        print(f"{[{counter}]} Translation:      {trs_text}")

        whisper_dict = {}
        whisper_dict['main_lang'] = detect_lang
        whisper_dict['main_lang_prob'] = detect_prob
        whisper_dict['text'] = src_text
        whisper_dict['translated_text'] = trs_text

        res_dict[f'stt_{counter}'] = whisper_dict


        start_second = end_second
        end_second = end_second + 30
        if start_second >= full_duration:
            break
    
    # TODO - shiry: detach whisper_model from GPU

def preprocess(audio):


    audioFile     = AudioSegment.from_mp3(audio)
    audioFile      = audioFile.set_frame_rate(16000)
    new_audio_name = os.path.join(os.path.dirname(audio), os.path.basename(audio).replace(".mp3", "_resampled.mp3"))
    audioFile.export(out_f=new_audio_name, format="mp3")
    print("File name: ", new_audio_name)

    #
    # waveform, sample_rate = torchaudio.load(audio)
    #
    # if waveform.shape[0] > 1:
    #     # Convert to mono by averaging channels
    #     waveform = waveform.mean(dim=0, keepdim=True)
    #
    # # Resample the audio to 16kHz
    # resampler = Resample(orig_freq=sample_rate, new_freq=16000)
    # resampled_waveform = resampler(waveform)
    #
    # new_audio = os.path.join(os.path.dirname(audio), os.path.basename(audio).replace(".mp3", "_resampled.mp3"))
    # print("File name: ", new_audio)
    # # Save the resampled audio to a new file
    # torchaudio.save(new_audio, resampled_waveform, 16000)
    # return new_audio
    return new_audio_name

def detect_lang(audio_file):
    model = whisper.load_model(whisper_model_size)
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    model = model.to(DEVICE)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    mel = mel.to(DEVICE)
    _, probs = model.detect_language(mel)
    # lang = max(probs, key=probs.get)
    # Sort the dictionary by value in descending order
    sorted_lang_prob = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    # Take the top 3 most probable languages
    top_3_lang = sorted_lang_prob[:3]
    print("Top 3 most probable languages:", top_3_lang)
    # print(f"Detected language: {lang}")
    return top_3_lang

def split_audio(audio_file, segment_length=5000, overlap=2000, segments_folder = ""):

    audio_file = preprocess(audio_file)

    segments_folder = os.path.basename(audio_file).replace(".mp3","")
    #segments_folder = f"horror/{segments_folder}"
    folder_name     = os.path.basename(audio_file).replace(".mp3","")
    segments_folder = f"{os.path.dirname(audio_file)}/{folder_name}"
    if not os.path.exists(segments_folder):
        os.mkdir(segments_folder)

    if not os.path.exists(segments_folder):
        os.makedirs(segments_folder)

    audio = AudioSegment.from_file(audio_file, format="mp3")

    duration_in_seconds = len(audio) / 1000.0  # pydub works in milliseconds
    print(f"duration_in_seconds: {duration_in_seconds}")
    
    segments = []
    segment_times = []  # In milliseconds
    segment_times_seconds = []  # In seconds
    new_files = []

    for i in range(0, len(audio), segment_length - overlap):
        segment = audio[i: i + segment_length]
        segments.append(segment)
        segment_times.append((i, i + segment_length))

        # Calculate start and end time in seconds
        start_time_seconds = i / 1000.0
        end_time_seconds = min((i + segment_length) / 1000.0, duration_in_seconds)
        segment_times_seconds.append(f"{start_time_seconds} seconds - {end_time_seconds} seconds")

        # Save to temporary mp3 file
        new_seg_audio_file = os.path.join(segments_folder, f"seg_{i}.mp3")
        new_files.append(new_seg_audio_file)
        segment.export(new_seg_audio_file, format="mp3")

    return segments, segment_times, segment_times_seconds, new_files

def process_segments(segment_files, segment_times_seconds, res_dict):
    
    feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    event_detection_model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    event_detection_model = event_detection_model.to(DEVICE)

    for i,audio_file in enumerate(segment_files):
        print("--------------------")
        audio_dict = {}
        print(os.path.basename(audio_file))
        # print(segment_times[i])
        print(segment_times_seconds[i])
        audio_dict['segment_length'] = segment_times_seconds[i]

        lang = detect_lang(audio_file)
        print(f"The detected language is: {lang}")
        audio_dict['lang'] = lang

        signal, sampling_rate = torchaudio.load(audio_file)
        
        signal = torchaudio.functional.resample(signal, orig_freq=sampling_rate, new_freq=TARGET_SR)

        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        # signal = signal.to(DEVICE)
        signal = signal.squeeze(0)

        # audio file is decoded on the fly
        inputs = feature_extractor(signal, sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        with torch.no_grad():
            logits = event_detection_model(**inputs).logits

        predicted_class_ids = torch.argmax(logits, dim=-1).item()
        
        predicted_label = event_detection_model.config.id2label[predicted_class_ids]

        k = 5
        print(f"Top {k} events detected in segment:")
        topk_values, topk_class_ids = torch.topk(logits, k=k, dim=-1)
        class_probs = torch.sigmoid(topk_values)
        class_probs = class_probs.detach().cpu().numpy()[0]
        topk_class_ids_list = topk_class_ids.tolist()[0]
        for ii, id in enumerate(topk_class_ids_list):
            predicted_label = event_detection_model.config.id2label[id]
            curr_prob = float(f"{class_probs[ii]:.3f}")
            print(f"{ii}: {predicted_label} ({curr_prob})")
            audio_dict[f'top_event_{ii+1}'] = predicted_label
            audio_dict[f'top_event_{ii+1}_prob'] = curr_prob

        res_dict[f'audio_segment_{i+1}'] = audio_dict

def translate(l_arabic):
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

    translator = pipeline("translation",
                          model=model,
                          tokenizer=tokenizer,
                          src_lang="arz_Arab",
                          tgt_lang="eng_Latn",
                          max_length=400)

    translate_res = translator(l_arabic)
    l_output      = []
    for output in translate_res:
        l_output.append(output['translation_text'])

    return l_output

def run_OCR(video_file_name, res_dict):


    ocr_dict = {}
    #res_dict['ocr'] = ""
    def filter_text_unicode(text):
        return ''.join(c for c in text if unicodedata.category(c).startswith(('L', 'P', 'N', 'Z')))

    VALID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    "אבגדהוזחטיכלמנסעפצקרשתךםןףץ "
                    "أبتثجحخدذرزسشصضطظعغفقكلمنهويىةء ")

    def filter_text_predefined(text):
        return ''.join(c for c in text if c in VALID_CHARS)

    videos_folder = os.path.dirname(video_file_name)
    
    # Open the video
    cap = cv2.VideoCapture(video_file_name)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0
    frames_folder = os.path.join(videos_folder, "frames")

    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)

    os.mkdir(frames_folder)
    frame_files = []
    while True:
        ret, frame = cap.read()
        
        # Break the loop if the video has ended
        if not ret:
            break
        
        # Save each frame as an image
        frame_file = os.path.join(frames_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_file, frame)
        frame_files.append(frame_file)
        frame_count += 1

    # Release the video capture object
    cap.release()

    frame_files_subset = frame_files[::4]

    print(f"Extracted {frame_count} frames.")
    reader = easyocr.Reader(['ar']) # this needs to run only once to load the model into memory
    texts = []
    text_frames = []
    prev_text = "" 
    for f in tqdm(frame_files_subset):
        results = reader.readtext(f)
        frame_text = ""
        for r in results:
            frame_text += filter_text_predefined(r[1]) + " "
        if prev_text.strip() != frame_text.strip() and Levenshtein.distance(prev_text.strip(), frame_text.strip()) > 5:
            print(frame_text)
            prev_text = frame_text
            texts.append(frame_text)
            frame = re.findall(r'-?\d+', os.path.basename(f))
            text_frames.append(frame[0])

    # trasnalte
    l_translated = translate(texts)

    text_df = pd.DataFrame()
    text_df['text'] = texts
    text_df['frame'] = text_frames
    text_df['translate'] = l_translated

    ocr_dict = {text_frames[i]: (texts[i], l_translated[i]) for i in range(len(text_frames))}
    #ocr_dict         = dict(zip(text_frames, texts, l_translated))
    # text_df.to_csv(f"ocr.csv")
    text_df.to_excel('ocr.xlsx', index=False, engine='openpyxl')
    res_dict['ocr'] = ocr_dict

def process_video(video_file):

    res_dict = {}

    audio_file = video_file.replace(".mp4", ".mp3")
    res_dict['file_name'] = audio_file
    # create an mp3 file with the audio
    extract_audio_from_video(video_file, audio_file)

    print("---------------------------------------------------------------")
    print("Entire audio language detection, transcription and translation:")
    # extract the full transcription with unknown language
    get_stt(audio_file, res_dict)

    # Split the audio to segments 
    segments, segment_times, segment_times_seconds, segment_files = split_audio(audio_file)

    print("---------------------------------------------------------------")
    print("Language Detection and Audio events per segment:")
    # Run language detection and event detection on each segment
    process_segments(segment_files, segment_times_seconds, res_dict)

    print("---------------------------------------------------------------")
    # Run OCR on the video file
    run_OCR(video_file, res_dict)

    # return results
    return res_dict

def pretty_print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            pretty_print_dict(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")

if __name__ == '__main__': 
    res_dict = process_video(video_file)
    pretty_print_dict(res_dict)

    

    


