import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Define base URL and headers
BASE_URL = "https://www.si.com/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def get_page_content(url):
    """Fetches HTML content of a webpage."""
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to retrieve {url} with status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


def extract_articles(content):
    """Extracts articles from the HTML content."""
    soup = BeautifulSoup(content, "lxml")
    articles = []

    # Find article elements
    for article in soup.find_all("article", class_="style_amey2v-o_O-wrapper_1wgo221"):
        try:
            title = article.find("h3").text.strip()
            message = article.find("p").text.strip()
            link = article.find("a")["href"]
            category = article.find("h6").text.strip()
            articles.append({"title": title, "message": message, "link": link, "category": category})
        except AttributeError:
            continue  # Skip articles missing expected data

    return articles


def scrape_sports_illustrated(pages=5):
    """Scrapes multiple pages of articles."""
    all_articles = []
    for page in range(1, pages + 1):
        url = f"{BASE_URL}/?page={page}"  # Adjust pagination URL based on site structure
        print(f"Scraping page: {url}")
        content = get_page_content(url)
        if content:
            articles = extract_articles(content)
            all_articles.extend(articles)
        time.sleep(1)  # Add delay to prevent overloading the server

    return all_articles


def save_to_csv(data, filename="sports_illustrated_articles.csv"):
    """Saves the scraped data to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


# Run the scraper
if __name__ == "__main__":
    articles = scrape_sports_illustrated(pages=10)  # Adjust number of pages as needed
    if articles:
        save_to_csv(articles)
