from services.transformer_services import (
    embedding_sentences,
    sentiment_sentences,
    summary_sentences,
)

sentences = [
    "雰囲気がとてもいいお店でした。また来店します",
    "味はいいけど店内の雰囲気はあまりよくなかったです",
]

model_name = "matsuo-lab/weblab-10b"
# model_name = "sonoisa/t5-base-japanese"
x = summary_sentences(
    sentences,
    model_name,
)
print(x)

y = embedding_sentences(
    sentences,
    "sonoisa/t5-base-japanese",
)
print(y)

# model_name = "matsuo-lab/weblab-10b"
# model_name = "sonoisa/t5-base-japanese"
model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
z = sentiment_sentences(
    sentences,
    model_name,
)
print(z)
