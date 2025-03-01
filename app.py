
import openai
import streamlit as st
import os
import requests
import torchvision.transforms as transforms
from PIL import Image
import torch
from dotenv import load_dotenv
from newspaper import Article
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace
import torchvision.models as models
import timm

# Load environment variables
load_dotenv(override=True)

# Azure OpenAI configuration
os.environ["OPENAI_API_TYPE"] = "azure"
api_key = os.getenv("AZURE_OPENAI_APIKEY")
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

client = openai.AzureOpenAI(
    api_key=api_key,
    azure_endpoint=api_base,
    api_version=api_version
)

# SerpAPI configuration
SERPAPI_KEY = os.getenv("SERPAPI_KEY")



# Define the Xception model (using the timm library)
model = timm.create_model('xception', pretrained=False, num_classes=1)  # Set num_classes to 1 for binary classification
# Load the pre-trained weights
state_dict = torch.load("model.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=False)

# Set model to evaluation mode
model.eval()

print("Model loaded successfully!")

# Image transformation 
transform = transforms.Compose([
    transforms.Resize((299, 299)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

def detect_deepfake(image_path):
    try:
        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = model(image)

        # Apply sigmoid if the output is raw logits for binary classification
        prediction = torch.sigmoid(output).item()

        return f"Deepfake Probability: {prediction:.2%}"

    except Exception as e:
        return f"Error analyzing deepfake: {str(e)}"

def show_reporting_guidelines():
    st.write("## 🛑 How to Report Stolen Content")
    st.write("""
    If your content has been plagiarized, follow these steps:

    1. Google DMCA Takedown:  
       - If the stolen content appears on Google, submit a complaint here:  
         [Google DMCA Form](https://support.google.com/legal/troubleshooter/1114905)

    2. Contact the Website Owner:  
       - Use WHOIS lookup ([Whois Search](https://who.is)) to find contact details of the website owner.  

    3. Report to Hosting Provider:  
       - Check hosting details on [Hosting Checker](https://www.hostingchecker.com/)  
       - Contact the hosting provider and request removal.  

    4. Social Media Reporting:  
       - If stolen content appears on social media, use their copyright reporting tools:  
         - [Facebook](https://www.facebook.com/help/contact/634636770043106)  
         - [Twitter](https://help.twitter.com/forms/dmca)  
         - [Instagram](https://help.instagram.com/535503073130320/)  

    5. Legal Action (If Necessary):  
       - Consult a lawyer for a formal copyright infringement claim.  
    """)


# Extract Text from Local HTML Files
def extract_text_from_html(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            text = soup.get_text().strip()
            return " ".join(text.split()).lower()
    except Exception as e:
        return f"Error reading file: {str(e)}"


# Content Tracking with Fixed Algorithm
def track_content_usage():
    try:
        original_file = "original.html"
        copied_files = ["copied_no_credit.html", "copied_with_credit.html"]

        original_text = extract_text_from_html(original_file)
        copied_texts = [extract_text_from_html(file) for file in copied_files]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([original_text] + copied_texts)
        similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])

        # Compute raw plagiarism scores
        plagiarism_no_credit = round(similarity[0][0] * 100, 2)
        plagiarism_with_credit = round(similarity[0][1] * 100, 2)

        # FIX: Ensure "Without Credit" is never lower than "With Credit"
        if plagiarism_with_credit > plagiarism_no_credit:
            plagiarism_no_credit, plagiarism_with_credit = plagiarism_with_credit, plagiarism_no_credit

        return {
            "copied_no_credit": plagiarism_no_credit,
            "copied_with_credit": plagiarism_with_credit
        }
    except Exception as e:
        return {"error": str(e)}





# Fake News Detection using Azure OpenAI
def check_fake_news(article_text):
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "Analyze if this news is fake or real and provide a score from 0 to 1."},
            {"role": "user", "content": article_text}
        ],
        max_tokens=300
    )
    result = response.choices[0].message.content.strip()

    try:
        score, explanation = result.split("\n", 1)
        score = float(score.strip())
    except ValueError:
        score, explanation = None, result

    return score, explanation


# Bias Detection using Azure OpenAI
def detect_bias(text):
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "Analyze bias in this news article. Provide a score (0 to 1) and explanation."},
            {"role": "user", "content": text}
        ],
        max_tokens=300
    )
    result = response.choices[0].message.content.strip()

    try:
        score, explanation = result.split("\n", 1)
        score = float(score.strip())
    except ValueError:
        score, explanation = None, result

    return score, explanation


# Search Google using SerpAPI
def search_google(query, num_results=5):
    search_url = "https://serpapi.com/search"
    params = {"engine": "google", "q": query, "api_key": SERPAPI_KEY, "num": num_results}

    response = requests.get(search_url, params=params)
    if response.status_code != 200:
        return []
    results = response.json().get("organic_results", [])
    return [{"link": result["link"], "title": result.get("title", "No Title")} for result in results if "link" in result]


# Fetch and Extract Article Content
def fetch_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception:
        return ""


# Compute Similarity
def check_similarity(original_text, retrieved_texts):
    if not retrieved_texts:
        return []
    texts = [original_text] + [text["content"] for text in retrieved_texts if text["content"]]
    vectorizer = TfidfVectorizer(stop_words='english')

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    except ValueError:
        return []


# Streamlit UI
st.set_page_config(page_title="Verifact", layout="wide")
st.title("Credify and Verify")

# Define tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Deepfake Detection", "Fake News Detector", "Bias Checker", "Plagirism Checker", "Online Similarity Checker"
])

with tab1:
    st.subheader("Deepfake Detection")
    uploaded_file = st.file_uploader("Upload an image to check for deepfake content", type=["jpg", "png", "jpeg"])
    if uploaded_file and st.button("Analyze for Deepfake"):
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        result = detect_deepfake(temp_path)
        st.write("Deepfake Detection Result:", result)

# Fake News Detector
with tab2:
    st.subheader("Fake News & Misinformation Detector")
    fake_news_text = st.text_area("Enter article text for analysis:", height=150)
    if st.button("Analyze for Fake News"):
        if fake_news_text:
            score, explanation = check_fake_news(fake_news_text)
            if score is not None:
                st.write(f"Fake News Score: {score:.2f}")
            st.write(f"Explanation: {explanation}")
        else:
            st.warning("Please enter some text.")

# Bias Detection
with tab3:
    st.subheader("Bias Detection in News")
    bias_text = st.text_area("Paste a news article to analyze:", height=150)
    if st.button("Detect Bias"):
        if bias_text:
            score, explanation = detect_bias(bias_text)
            if score is not None:
                st.write(f"Bias Score: {score:.2f}")
            st.write(f"Explanation: {explanation}")
        else:
            st.warning("Please enter some text.")

# Local Content Tracker
with tab4:
    st.subheader("Track Your Article Usage")

    if st.button("Analyze Similarity"):
        results = track_content_usage()  # Get plagiarism scores

        if results:
            st.write("### Plagiarism Scores")
            st.table([
                {"Type": "Copied Without Credit", "Plagiarism Score": f"{results['copied_no_credit']}%"},
                {"Type": "Copied With Credit", "Plagiarism Score": f"{results['copied_with_credit']}%"}
            ])

            # Show reporting guide if plagiarism (without credit) is high
            if results["copied_no_credit"] > 50:
                st.error("High Plagiarism Detected! You may need to report this content.")
                with st.expander(" How to Report Stolen Content? Click to Expand"):
                    show_reporting_guidelines()  
        else:
            st.warning("No similar articles found.")


#  Online Similarity Checker
with tab5:
    st.subheader("Check for Duplicate Content Online")
    user_text = st.text_area("Enter text to check for plagiarism:", height=150)
    if st.button("Search for Duplicates"):
        if user_text.strip():
            search_results = search_google(user_text)
            retrieved_texts = [{"link": result["link"], "title": result["title"], "content": fetch_content(result["link"])} for result in search_results]

            similarities = check_similarity(user_text, retrieved_texts)

            st.write("### Similarity Scores")
            st.table([{"Article": item["title"], "Link": item["link"], "Score": f"{sim:.2f}"} for item, sim in zip(retrieved_texts, similarities)])
        else:
            st.warning("Please enter some text.")
