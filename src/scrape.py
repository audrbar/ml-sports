import os
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from utils import append_csv, find_unique_values, DATA_DIR

# Define URL and headers
SI_URL = "https://www.si.com/"
SUB_URLS = [
    '', 'nfl', 'nba', 'olympics', 'college/college-basketball', 'college/college-football',
    'nhl', 'mlb', 'wnba', 'mma', 'soccer', 'golf', 'tennis', 'boxing', 'racing', 'wrestling',
    'high-school', 'super-bowl'
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/91.0.4472.124 Safari/537.36"
}

# Define paths for storing scraped data
ARTICLES_LINKS_SCRAPED_CSV = os.path.join(DATA_DIR, "articles_links_scraped.csv")
ARTICLES_LINKS_SCRAPED_TODAY_CSV = os.path.join(DATA_DIR, "articles_links_scraped_today.csv")
ARTICLES_LINKS_UNIQUE_CSV = os.path.join(DATA_DIR, "articles_links_unique.csv")
ARTICLES_SCRAPED_CSV = os.path.join(DATA_DIR, "articles_scraped.csv")


# Fetch HTML content
def get_page_content(url):
    """Fetches HTML content from a webpage with error handling."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()  # Raise exception for 4xx/5xx errors
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


# Scrape article links
def scrape_article_links(base_url, sub_urls):
    """Scrapes article links from different subcategories of a website."""
    article_links_set = set()  # Use a set to avoid duplicates

    for sub_url in sub_urls:
        url = f"{base_url}{sub_url}"  # Construct full URL
        print(f"Scraping links from: {url}")

        content = get_page_content(url)
        if content:
            soup = BeautifulSoup(content, "lxml")

            for article in soup.find_all("article"):
                link_tag = article.find("a", href=True)
                if link_tag:
                    full_link = link_tag["href"]
                    if full_link.startswith("/"):
                        full_link = base_url + full_link  # Convert relative links to absolute
                    article_links_set.add(full_link)

        time.sleep(random.uniform(2, 5))  # Random delay to prevent detection

    return pd.DataFrame(list(article_links_set), columns=["url"])


# Scrape full article content
def scrape_articles_content(links_file, sleep_time=2):
    """Reads article links from a CSV file, fetches pages, and scrapes <h1> and <p> tags."""
    if not os.path.exists(links_file):
        print(f"Error: {links_file} not found.")
        return []

    df = pd.read_csv(links_file)

    if "url" not in df.columns:
        print("Error: CSV file must contain a 'url' column.")
        return []

    articles_data = []

    for index, row in df.iterrows():
        url = row["url"]
        print(f"Scraping article {index + 1}: {url}")

        try:
            content = get_page_content(url)
            if not content:
                continue

            soup = BeautifulSoup(content, "lxml")

            # Extract <h1> and <p> content
            title = " ".join([h1.text.strip() for h1 in soup.find_all("h1") if h1.text.strip()])
            content = " ".join([p.text.strip() for p in soup.find_all("p") if p.text.strip()])

            # Store article data
            articles_data.append({
                "url": url,
                "title": title,
                "content": content
            })

            print(f"Scraped: {title[:150]}...")  # Show preview of title
            time.sleep(random.uniform(sleep_time, sleep_time + 3))  # Random delay to prevent detection

        except requests.exceptions.RequestException as e:
            print(f"Error opening {url}: {e}")

    return articles_data


# Run the scraper
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)  # Ensure data directory exists

    # # Step 1: Scrape article links
    # article_links = scrape_article_links(SI_URL, SUB_URLS)
    # append_csv(article_links, LINKS_CSV)

    # Find unique values
    # find_unique_values("articles_links.csv", LINKS_CSV, LINKS_UNIQUE_CSV)

    # Step 2: Scrape article content
    scraped_articles = scrape_articles_content(ARTICLES_LINKS_UNIQUE_CSV)
    append_csv(scraped_articles, ARTICLES_SCRAPED_CSV)
    print(f"\nScraping completed! All data saved to {scraped_articles}.")
