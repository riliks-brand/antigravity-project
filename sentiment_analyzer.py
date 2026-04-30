"""
Sentiment Analyzer — Phase 4
==============================
Lightweight NLP module designed specifically for Forex.
Fetches live RSS feeds (e.g., ForexLive) and applies a keyword-based
scoring system to measure market sentiment (Hawkish/Dovish).

Returns a sentiment score from -1.0 to +1.0 for major currencies.
"""

import requests
import xml.etree.ElementTree as ET
import re
import logging
from datetime import datetime
from config import Config

logger = logging.getLogger("Sentiment")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(getattr(Config, "LOG_FILE", "bot.log"), encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("\033[96m%(asctime)s\033[0m [%(levelname)s] %(message)s"))
    logger.addHandler(_ch)

# Dictionary of Forex-specific keywords and their sentiment weight (focusing on USD)
# Positive = USD Bullish (Hawkish / Good economy)
# Negative = USD Bearish (Dovish / Bad economy)
USD_SENTIMENT_DICT = {
    "hawkish": 0.8,
    "dovish": -0.8,
    "rate hike": 0.6,
    "rate cut": -0.8,
    "inflation rises": 0.5,
    "inflation falls": -0.5,
    "cpi beats": 0.5,
    "cpi misses": -0.5,
    "strong jobs": 0.5,
    "weak jobs": -0.5,
    "nfp beats": 0.6,
    "nfp misses": -0.6,
    "recession": -0.7,
    "fed hikes": 0.8,
    "fed cuts": -0.8,
    "powell hawkish": 0.7,
    "powell dovish": -0.7,
    "gdp growth": 0.4,
    "gdp contraction": -0.5,
    "retail sales beat": 0.4,
    "retail sales miss": -0.4,
    "safe haven demand": 0.4,  # Usually boosts USD and Gold
    "risk off": 0.3,
    "risk on": -0.3,
}

# Supported RSS Feeds
RSS_FEEDS = [
    "https://www.forexlive.com/feed/news",
    # Add more as needed, e.g., DailyFX or Investing.com
]

def clean_text(text):
    """Clean HTML and extra spaces from RSS descriptions."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
    return text.lower().strip()

def analyze_sentiment_usd():
    """
    Fetches the latest forex news and calculates a USD sentiment score.
    Returns:
        float: Sentiment score between -1.0 and 1.0
    """
    total_score = 0.0
    valid_articles = 0
    max_articles = 15  # Only check the latest 15 articles to ensure relevance

    for feed_url in RSS_FEEDS:
        try:
            response = requests.get(feed_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code != 200:
                logger.warning(f"[Sentiment] Failed to fetch RSS: {feed_url}")
                continue

            root = ET.fromstring(response.content)
            
            # Parse RSS 2.0 format
            for item in root.findall('.//item')[:max_articles]:
                title = item.find('title')
                desc = item.find('description')
                
                title_text = clean_text(title.text) if title is not None else ""
                desc_text = clean_text(desc.text) if desc is not None else ""
                
                combined_text = title_text + " " + desc_text
                
                article_score = 0.0
                matched_keywords = []

                for phrase, weight in USD_SENTIMENT_DICT.items():
                    if phrase in combined_text:
                        article_score += weight
                        matched_keywords.append(phrase)

                # Cap individual article score
                if article_score > 0:
                    article_score = min(article_score, 1.0)
                elif article_score < 0:
                    article_score = max(article_score, -1.0)

                if matched_keywords:
                    total_score += article_score
                    valid_articles += 1
                    logger.debug(f"[Sentiment] Match: {matched_keywords} -> Score: {article_score:+.2f} | '{title_text[:50]}...'")

        except Exception as e:
            logger.error(f"[Sentiment] Error parsing {feed_url}: {e}")

    if valid_articles == 0:
        return 0.0

    # Average score across relevant articles
    avg_score = total_score / valid_articles
    
    # Optional: Apply an decay or boost if there's a strong consensus
    if valid_articles >= 3 and abs(avg_score) > 0.3:
        avg_score *= 1.2  # Consensus boost

    # Final clip
    final_score = max(min(avg_score, 1.0), -1.0)
    
    if abs(final_score) >= 0.2:
        logger.info(f"[Sentiment] Current USD Sentiment: {final_score:+.2f} (based on {valid_articles} recent news items)")
        
    return final_score

if __name__ == "__main__":
    print("Fetching live Forex sentiment...")
    score = analyze_sentiment_usd()
    print(f"Final USD Sentiment Score: {score:+.2f}")
