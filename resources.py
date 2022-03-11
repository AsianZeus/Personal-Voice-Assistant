from config import *
import speech_recognition as sr
from speechbrain.pretrained import EncoderClassifier
from transformers.utils import logging as logging_transformers
logging_transformers.get_logger().setLevel(logging_transformers.CRITICAL)
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, BlenderbotForConditionalGeneration, BlenderbotTokenizer, AutoTokenizer,AutoModelForSeq2SeqLM
import pickle
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from transformers import pipeline

id2label = pickle.load(open(id2label_path, 'rb'))

eng_processor = Wav2Vec2Processor.from_pretrained(eng_model_path)
eng_model = Wav2Vec2ForCTC.from_pretrained(eng_model_path)

hi_processor = Wav2Vec2Processor.from_pretrained(hi_model_path)
hi_model = Wav2Vec2ForCTC.from_pretrained(hi_model_path)

speech_models, cfg, task = load_model_ensemble_and_task(tts_model_path, arg_overrides={"config_yaml": "./config.yaml", "data": tts_path, "vocoder": "hifigan", "fp16": False})
speech_generator = task.build_generator(speech_models, cfg)

mt_tokenizer = AutoTokenizer.from_pretrained(mt_model_path)
mt_model = AutoModelForSeq2SeqLM.from_pretrained(mt_model_path)

conversational_tokenizer = BlenderbotTokenizer.from_pretrained(coversational_model_path)
conversational_model = BlenderbotForConditionalGeneration.from_pretrained(coversational_model_path)

speech_classifier = EncoderClassifier.from_hparams(source=speech_classifier_path)

restore_punct_model = pipeline("ner", model=restore_punct_model_path, tokenizer=restore_punct_model_path, aggregation_strategy="average")

recorder = sr.Recognizer()