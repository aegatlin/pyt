from datasets import Audio, load_dataset
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
# print(encoding)

pt_batch = tokenizer(
    [
        "We are very happy to show you the 🤗 Transformers library.",
        "We hope you don't hate it.",
    ],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

# print(pt_batch)

pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)

pt_outputs = pt_model(**pt_batch)
# print(pt_outputs)
pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(pt_predictions)

pt_save_directory = "./pt_save_pretrained"
tokenizer.save_pretrained(pt_save_directory)
pt_model.save_pretrained(pt_save_directory)

# model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
#
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
# res = classifier(
#     "Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers."
# )
#
# print(res)

# speech_recognizer = pipeline(
#     "automatic-speech-recognition", model="facebook/wav2vec2-base-960h"
# )
#
# dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
# dataset = dataset.cast_column(
#     "audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate)
# )
#
# result = speech_recognizer(dataset[:4]["audio"])
# print([d["text"] for d in result])
