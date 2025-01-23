import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Define URL and headers
SI_URL = "https://www.si.com/"
SUB_URLS = ['', 'nfl', 'nba', 'wnba', 'college/college-basketball', 'college/college-football', 'mlb', 'soccer', 'golf']
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/91.0.4472.124 Safari/537.36"
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
    articles_dict = []

    # Find article elements
    for article in soup.find_all("article"):
        try:
            title = article.find("h3").text.strip()
            intro = article.find("p").text.strip()
            author = article.find("h6").text.strip()
            link = article.find("a")["href"]
            articles_dict.append({"title": title, "intro": intro, "author": author, "link": link})
        except AttributeError:
            continue  # Skip articles missing expected data

    return articles_dict


def scrape_web_page(base_url, sub_urls):
    """Scrapes multiple pages of articles."""
    all_articles = []
    for sub_url in sub_urls:
        url = f"{base_url}{sub_url}"  # Adjust pagination URL based on site structure
        print(f"Scraping page: {url}")
        content = get_page_content(url)
        if content:
            articles_html = extract_articles(content)
            all_articles.extend(articles_html)
        time.sleep(2)  # Add delay to prevent overloading the server

    return all_articles


def append_csv(data, filename="articles_scraped.csv"):
    """Appends the scraped data to a CSV file."""
    # Check if the file already exists
    file_exists = os.path.isfile(filename)

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Append to the file if it exists, otherwise create it
    if file_exists:
        df.to_csv(filename, mode='a', index=False, header=False)
        print(f"Data appended to {filename}")
    else:
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")


# Run the scraper
if __name__ == "__main__":
    articles = scrape_web_page(SI_URL, SUB_URLS)
    if articles:
        append_csv(articles)
