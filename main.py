## Author : Akshat Surolia
## Date: March 2022
## Description: This is the main file for the Voice Assistant.
import logging
logging.getLogger('fairseq').setLevel(logging.CRITICAL)
logging.getLogger('speechbrain').setLevel(logging.CRITICAL)
import speech_recognition as sr
import os
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from resources import *
from config import *
from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
conversation = Conversation(conversational_tokenizer, conversational_model)

os.system('cls' if os.name == 'nt' else 'clear')

with sr.Microphone(sample_rate=16000) as source:
    print("Calibrating microphone...")
    audio = recorder.adjust_for_ambient_noise(source, duration=5) # listen for 1 second to calibrate the energy threshold for ambient noise levels
    print("Listening...")
    transcription = ""
    while True:
        audio_data = get_audio_data(recorder, source) # listens and extract wave stream of audio data
        prediction =  speech_classifier.classify_batch(audio_data) # language classifier
        
        en = 20 # laguage index for english
        hi = 35 # laguage index for hindi
        sc = prediction[0][0]

        # if sc[hi]<sc[en]:
        if sc[hi]-sc[en]<0.2:
            transcription = get_transcription(audio_data, eng_processor, eng_model) # speech to text
        else:
            transcription = get_transcription(audio_data, hi_processor, hi_model) # speech to text
            transcription = get_translation(transcription, mt_tokenizer, mt_model) # translating hindi to english

        if "stop" in transcription: #stopping condition
                break
        if transcription:
            transcription = restore_punctuation(transcription, restore_punct_model) # restoring punctuation
            print(f"{'>> You:':{' '}{'<'}{10}}",transcription)
            response = conversation.get_conversation_response(transcription) # getting response from assistant
            print(f"{'>> Alice:':{' '}{'<'}{10}}", response)
            synthesize(task, speech_models, speech_generator, TTSHubInterface, text=response) # synthesizing response in speech

