from transformers import pipeline
import requests

def check_fake_news(article_text):
    model = pipeline("text-classification", model="microsoft/deberta-v3-large-mnli")
    labels = ["real", "fake"]
    result = model(article_text, labels)
    return {"fake_news_score": result}
