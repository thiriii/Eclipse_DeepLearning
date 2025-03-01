# Eclipse_DeepLearning
Deep Learning Week

Verifact

Verifact is a powerful tool for detecting deepfake images, fake news, biased content, plagiarism, and online article similarities. It uses AI models and various advanced algorithms to help users identify misinformation, detect manipulated media, and ensure the authenticity of content. This tool also provides guidelines for reporting stolen content and tracking plagiarism across the web.

Features
	•	Deepfake Detection: Upload an image to check for deepfake content using the Xception model.
	•	Fake News Detector: Analyze news articles for potential misinformation using Azure OpenAI’s language models.
	•	Bias Checker: Evaluate the potential bias in news articles and get a score along with an explanation.
	•	Plagiarism Checker: Check for plagiarism within your local content and detect copied content with and without credit.
	•	Online Similarity Checker: Search for duplicate content online by entering text and comparing it with articles retrieved from Google using SerpAPI.

Technologies Used
	•	Azure OpenAI: For fake news and bias detection, leveraging language models.
	•	Xception (Deepfake Detection): A deep learning model used to classify deepfake content.
	•	TfidfVectorizer & Cosine Similarity: For content comparison and plagiarism detection.
	•	SerpAPI: To retrieve relevant search results from Google for the online similarity checker.
	•	Streamlit: For creating an interactive web app interface.
	•	Torch & Timmy: For running deep learning models like Xception for deepfake detection.
