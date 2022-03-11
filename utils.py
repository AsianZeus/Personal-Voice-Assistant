import torch
import io
from pydub import AudioSegment
import numpy as np
import nltk
import simpleaudio as sa

def process_input(text):
    conj = ['although','because','that','though','unless','yet']
    processed_text_sequences = []
    paragraph = text.strip().replace("-", " ")
    sentences = nltk.sent_tokenize(paragraph)
    for sentence in sentences:
        phrases = sentence.split(",")
        for txt in phrases:
            txt = txt+"," if txt[-1].isalpha() else txt
            lst=[]
            a=1
            for word in txt.split():
                if word not in conj:
                    lst.append(word)
                elif (word in conj):
                    sent=' '.join(lst)
                    if sent:
                        processed_text_sequences.append(sent.strip())
                    lst.clear()
                    processed_text_sequences.append(word)
                if (a==len(txt.split())):
                    if(len(lst)):
                        sent=' '.join(lst)
                        if sent:
                            processed_text_sequences.append(sent.strip())
                a=a+1
    return processed_text_sequences

def synthesize(task, models,generator, TTSHubInterface, text="Hello there!"):
    wav = []
    paragraphs = text.split("\n")
    for paragraph in paragraphs:
        processed_text_sequences = process_input(paragraph)
        for phrase in processed_text_sequences:
            sample = TTSHubInterface.get_model_input(task, phrase)
            tempwav, rate = TTSHubInterface.get_prediction(task, models[0], generator, sample)
            tempwav = tempwav.numpy()
            if phrase[-1]==".":
                tempwav = np.append(tempwav, np.zeros(5000, dtype=np.float32))
            wav.extend(tempwav)
        wav.extend(np.zeros(10000, dtype=np.float32))
    wave_obj = sa.WaveObject(audio_data=np.array(wav), num_channels=1, sample_rate=rate, bytes_per_sample=4)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def get_audio_data(r, source):
    audio = r.listen(source, phrase_time_limit=15)
    data = io.BytesIO(audio.get_wav_data())
    clip = AudioSegment.from_file(data)
    x = torch.FloatTensor(clip.get_array_of_samples())
    return x

def get_transcription(x, processor, model):
    input_values = processor(x, sampling_rate=16000, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, axis=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    transcription = str(transcription).lower().strip()
    return transcription

def get_translation(transcription, mt_tokenizer, mt_model):
    model_inputs = mt_tokenizer(transcription, return_tensors="pt")
    generated_tokens = mt_model.generate(**model_inputs)
    transcription = mt_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    transcription = ' '.join(transcription).strip()
    return transcription

def restore_punctuation(text, model):
    ner_results = model(text)
    id2label = {'LABEL_0': 'OU','LABEL_1': 'OO','LABEL_2': '.O','LABEL_3': '!O','LABEL_4': ',O','LABEL_5': '.U', 'LABEL_6': '!U', 'LABEL_7': ',U', 'LABEL_8': ':O', 'LABEL_9': ';O', 'LABEL_10': ':U', 'LABEL_11': "'O", 'LABEL_12': '-O', 'LABEL_13': '?O', 'LABEL_14': '?U'}
    punct_resp =""
    for token in ner_results:
        entity = token['entity_group']
        word = token['word']
        label = id2label[entity]
        if label[-1] == "U":
            punct_wrd = word.capitalize()
        else:
            punct_wrd = word
        if label[0] != "O":
            punct_wrd += label[0]
        punct_resp += punct_wrd + " "
    punct_resp = punct_resp.strip()
    if punct_resp[-1].isalnum():
        punct_resp += "."
    punct_resp = punct_resp.strip().replace(" ' ", "'")
    return punct_resp

class Conversation:
    def __init__(self, conversational_tokenizer, conversational_model):
        self.conversational_tokenizer = conversational_tokenizer
        self.conversational_model = conversational_model
        self.chat_history_ids = None
        self.step = 0

    def get_conversation_response(self, transcription):
        input_ids = self.conversational_tokenizer.encode(transcription + self.conversational_tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([self.chat_history_ids, input_ids], dim=-1) if self.step > 0 else input_ids
        self.chat_history_ids = self.conversational_model.generate(
        input_ids,
        do_sample=True,
        min_length=bot_input_ids.shape[-1] + 2,
        max_length=1000,
        top_k=50,
        top_p=0.95,
        pad_token_id=self.conversational_tokenizer.eos_token_id)
        response = self.conversational_tokenizer.decode(self.chat_history_ids[0], skip_special_tokens=True)
        self.step += 1
        return response.strip()

    def reset(self):
        self.chat_history_ids = None
        self.step = 0
        self.transcription = None
        self.translation = None
        self.response = None
        self.conversation_response = None