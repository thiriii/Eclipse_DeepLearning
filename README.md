# Verifact

Verifact is a powerful tool for detecting deepfake images, fake news, biased content, plagiarism, and online article similarities. It uses AI models and various advanced algorithms to help users identify misinformation, detect manipulated media, and ensure the authenticity of content. This tool also provides guidelines for reporting stolen content and tracking plagiarism across the web.

## Features

- **Deepfake Detection**: Upload an image to check for deepfake content using the Xception model.
- **Fake News Detector**: Analyze news articles for potential misinformation using Azure OpenAI's language models.
- **Bias Checker**: Evaluate the potential bias in news articles and get a score along with an explanation.
- **Plagiarism Checker**: Check for plagiarism within your local content and detect copied content with and without credit.
- **Online Similarity Checker**: Search for duplicate content online by entering text and comparing it with articles retrieved from Google using SerpAPI.

## Technologies Used

- **Azure OpenAI**: For fake news and bias detection, leveraging language models.
- **Xception (Deepfake Detection)**: A deep learning model used to classify deepfake content.
- **TfidfVectorizer & Cosine Similarity**: For content comparison and plagiarism detection.
- **SerpAPI**: To retrieve relevant search results from Google for the online similarity checker.
- **Streamlit**: For creating an interactive web app interface.
- **Torch & Timmy**: For running deep learning models like Xception for deepfake detection.

## Requirements

- Python 3.7+
- openai
- streamlit
- torch
- timm
- requests
- newspaper3k
- sklearn
- deepface
- bs4
- python-dotenv

## Setup

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/yourusername/verifact.git
    cd verifact
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:

    Create a `.env` file in the root directory and add the following:

    ```bash
    AZURE_OPENAI_APIKEY=your_api_key
    AZURE_OPENAI_ENDPOINT=your_endpoint
    AZURE_OPENAI_API_VERSION=2024-02-15-preview
    AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
    SERPAPI_KEY=your_serpapi_key
    ```

4. Run the app:

    ```bash
    streamlit run app.py
    ```

## Features in Detail

### Deepfake Detection

Upload an image to check whether it is a deepfake. The image is processed using a deep learning model (Xception), and a probability score is provided for the likelihood that the image is a deepfake.

### Fake News Detector

Enter text from a news article to check if it contains misinformation. The tool uses Azure OpenAI to analyze the article and provide a score (0 to 1) along with an explanation.

### Bias Checker

Analyze a news article for potential bias. The tool uses Azure OpenAI to evaluate the text and provide a bias score, along with an explanation of the findings.

### Plagiarism Checker

Track your content and check for plagiarism by comparing it to other articles. The tool compares your original text with both credited and non-credited versions, and it provides a plagiarism score based on text similarity.

### Online Similarity Checker

Enter any text to check for similar articles online. The app uses SerpAPI to fetch relevant search results from Google and then analyzes the similarity between your text and the retrieved articles using TF-IDF and Cosine Similarity.

### Reporting Stolen Content

If your content is plagiarized, follow the guidelines provided within the app to take action:

- **DMCA Takedown**: Submit a complaint to Google.
- **Contact Website Owners**: Use WHOIS lookup to find contact details.
- **Report to Hosting Providers**: Contact the hosting provider to request removal.
- **Social Media Reporting**: Use social media's copyright tools to report stolen content.

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add your feature'`)
5. Push to the branch (`git push origin feature/your-feature`)
6. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
