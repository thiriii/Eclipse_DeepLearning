import requests
from bs4 import BeautifulSoup

def track_content_usage(article_url):
    try:
        response = requests.get(article_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()[:100]

        search_url = f"https://www.google.com/search?q={text}"
        headers = {"User-Agent": "Mozilla/5.0"}
        search_response = requests.get(search_url, headers=headers)
        search_soup = BeautifulSoup(search_response.text, 'html.parser')

        results = [link.get("href") for link in search_soup.find_all("a") if "url?q=" in link.get("href")]
        return {"found_on": results}
    except Exception as e:
        return {"error": str(e)}
