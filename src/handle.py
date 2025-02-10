import re
import os
import logging
from utils import adjust_pandas_display, load_csv, write_csv, DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define paths for storing data
ARTICLES_SCRAPED_CSV = os.path.join(DATA_DIR, "articles_scraped.csv")
ARTICLES_CLEANED_CSV = os.path.join(DATA_DIR, "articles_cleaned.csv")


def handle_values(data_frame, text_columns):
    """Preprocesses a DataFrame by handling missing values, removing duplicates,
    cleaning text columns, and reindexing."""
    logging.info("Handling missing values and duplicates...")
    df_cleaned = data_frame.dropna().drop_duplicates().reset_index(drop=True)

    for col in text_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].str.strip()

    logging.info(f"Data cleaned. {len(df_cleaned)} records remaining.")
    return df_cleaned


def category_column(data_frame, link_column='url', category_column_='category'):
    """Extracts categories from a link column and updates the DataFrame."""
    logging.info("Extracting categories from URLs...")
    data_frame[category_column_] = data_frame[link_column].apply(
        lambda url: re.search(r"https://www\.si\.com/([^/]+)/", url).group(1) if isinstance(url, str) and re.search(
            r"https://www\.si\.com/([^/]+)/", url) else None)
    data_frame.drop(columns=[link_column], inplace=True)
    return data_frame


def category_distribution(data_frame, categories_to_drop, target_column, min_count=5, max_samples=100):
    """Filters categories with fewer than min_count occurrences and applies sampling."""
    logging.info("Analyzing category distribution...")
    class_counts = data_frame[target_column].value_counts()
    logging.info(f"Initial class distribution:\n{class_counts}")

    rare_classes = class_counts[class_counts < min_count].index.tolist()

    if rare_classes:
        logging.info(f"Filtering out rare classes: {rare_classes}")
        filtered_data = data_frame[~data_frame[target_column].isin(rare_classes)]
    else:
        logging.info("No rare classes found.")
        filtered_data = data_frame

    # Drop specific categories
    filtered_data = filtered_data[~filtered_data[target_column].isin(categories_to_drop)]

    # Apply sampling to balance category counts
    sampled_data = filtered_data.groupby(target_column).apply(
        lambda x: x.sample(n=min(len(x), max_samples), random_state=42)).reset_index(drop=True)

    filtered_class_counts = sampled_data[target_column].value_counts()
    logging.info(f"Filtered and sampled class distribution:\n{filtered_class_counts}")
    logging.info(f"{len(sampled_data)} records remaining after filtering and sampling.")
    return sampled_data


if __name__ == "__main__":
    adjust_pandas_display(max_rows=None, max_columns=None, width=1000)

    df = load_csv(ARTICLES_SCRAPED_CSV)
    df_clean = handle_values(df, ['url', 'title', 'content'])
    df_category = category_column(df_clean)
    df_distribution = category_distribution(df_category, ["fannation", "onsi"],
                                            target_column="category", min_count=50, max_samples=170)

    write_csv(df_distribution, ARTICLES_CLEANED_CSV)
    logging.info("Data processing completed.")
