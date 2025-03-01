from transformers import pipeline

def detect_bias(text):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["left-wing bias", "right-wing bias", "neutral"]
    results = classifier(text, labels)
    return {"bias_analysis": results}
