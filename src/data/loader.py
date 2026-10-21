"""Data Loading, Validation, Sanitization, and Benchmark Generation Module.

Implements the specifications of Module A: Pre-processing.
"""

from typing import Optional, List, Dict
from datetime import datetime, timedelta
import re
import numpy as np
import pandas as pd


# Schema definition
REQUIRED_COLUMNS = ["text", "rating", "reviewer_id", "product_id", "date"]


def validate_schema(df: pd.DataFrame) -> bool:
    """Verifies that the DataFrame contains all mandatory columns."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    return True


def sanitize_text(text: str) -> str:
    """Sanitizes raw review text: removes HTML tags, normalizes whitespace and control chars."""
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    clean = re.sub(r"<[^>]+>", " ", text)
    # Strip control characters
    clean = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", clean)
    # Normalize multiple whitespaces
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def sanitize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Module A quality gates and sanitization filters:

    1. Validates schema
    2. Drops null texts/ratings
    3. Normalizes ratings to integer 1..5
    4. Parses dates into pandas datetime
    5. Cleans text and filters out reviews with fewer than 3 tokens
    6. Deduplicates exact duplicates
    """
    validate_schema(df)
    clean_df = df.copy()

    # Drop nulls in critical fields
    clean_df = clean_df.dropna(subset=["text", "rating", "reviewer_id", "product_id", "date"])

    # Ensure rating is integer 1 to 5
    clean_df["rating"] = pd.to_numeric(clean_df["rating"], errors="coerce")
    clean_df = clean_df.dropna(subset=["rating"])
    clean_df["rating"] = clean_df["rating"].astype(int)
    clean_df = clean_df[(clean_df["rating"] >= 1) & (clean_df["rating"] <= 5)]

    # Parse dates
    clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")
    clean_df = clean_df.dropna(subset=["date"])

    # Clean text
    clean_df["text"] = clean_df["text"].apply(sanitize_text)

    # Filter out reviews with fewer than 3 tokens
    token_counts = clean_df["text"].apply(lambda t: len(t.split()))
    clean_df = clean_df[token_counts >= 3]

    # Deduplicate exact duplicate reviews for same product by same reviewer
    clean_df = clean_df.drop_duplicates(subset=["reviewer_id", "product_id", "text"])

    # Sort chronologically
    clean_df = clean_df.sort_values(by="date").reset_index(drop=True)
    return clean_df


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Loads and sanitizes review dataset from CSV file."""
    df = pd.read_csv(csv_path)
    return sanitize_dataset(df)


def generate_benchmark_dataset(
    n_reviews: int = 3500,
    n_reviewers: int = 800,
    n_products: int = 250,
    fake_share: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generates a realistic e-commerce review dataset modeling both authentic
    human review behavior and fraudulent manipulation patterns (Yelp/Amazon distribution).

    Includes:
    - Reviewer burstiness patterns
    - Rating-text sentiment incongruity
    - Product surge campaigns
    - Copy-paste templates
    - AI-generated generic phrasing
    """
    rng = np.random.RandomState(random_seed)

    base_date = datetime(2025, 1, 1)

    # Reviewers & Products pools
    reviewers = [f"rev_{i:04d}" for i in range(n_reviewers)]
    products = [f"prod_{j:04d}" for j in range(n_products)]

    # 1. Rich vocabulary & combinatorial components for GENUINE reviews
    openers_pos = [
        "Purchased this item after reading several tech comparisons.",
        "Been using this model for about three months now in my daily routine.",
        "Got this as a replacement for an older unit that stopped working.",
        "Ordered this during the prime sale and it arrived in two days.",
        "Initial impressions after three weeks of regular hands-on usage:",
        "Decided to give this brand a try based on friend recommendations.",
        "Bought this specifically for an upcoming weekend camping trip.",
        "Unboxed and set it up within 20 minutes without any headaches.",
    ]

    features_pos = [
        "The battery life comfortably lasts two full days on a single charge.",
        "Build quality is solid with brushed aluminum casing and no panel flex.",
        "The display brightness is excellent even under direct sunlight.",
        "Audio clarity is crisp with well-balanced mids and clean trebles.",
        "The ergonomic grip fits nicely in the palm and does not slip.",
        "Fits true to size for a men's 10.5 with very supportive arch cushioning.",
        "Motor runs quiet under normal load and does not overheat.",
        "The zipper glides smoothly and the water-resistant coating works as advertised.",
        "The stainless steel blades slice through vegetables effortlessly.",
        "Firmware update installed seamlessly over Wi-Fi on first boot.",
    ]

    caveats_pos = [
        "The included power cord is somewhat short at only 3 feet, but easily manageable.",
        "A bit heavier than expected at 1.4 lbs, though that contributes to the sturdy feel.",
        "The companion mobile app has a slight learning curve for custom presets.",
        "Color in person is slightly darker than the product photos online.",
        "The cooling fan emits a faint hum at max speed, though barely noticeable.",
        "Packaging had a minor tear on the outer corner, but inside was well protected.",
    ]

    closers_pos = [
        "Overall very satisfied with the purchase and would buy again.",
        "Solid value for money, dependable performance so far.",
        "Highly recommended if you want reliable quality without overpaying.",
        "A great addition to my workspace setup. Glad I chose this model.",
        "Meets all my requirements and exceeded my expectations for the price.",
    ]

    # Genuine negative components (1-star and 2-star)
    openers_neg = [
        "Very disappointed with this purchase.",
        "Stopped working properly after just two weeks of gentle use.",
        "I rarely write negative reviews, but this item has serious quality issues.",
        "Unfortunately does not live up to the advertised claims.",
        "Expected much better reliability given the price tag.",
    ]

    complaints_neg = [
        "The battery drains from 100% to dead in less than two hours.",
        "The zipper split along the bottom seam during my first business trip.",
        "Screen flickers violently when connected via HDMI at 60Hz.",
        "Motor started emitting a sharp burning plastic smell after 10 minutes of use.",
        "The plastic latch on the lid snapped under normal thumb pressure.",
        "Missing the main mounting brackets and power adapter from the sealed box.",
        "Customer service took 9 days to respond and refused a warranty replacement.",
        "Way smaller than pictured, only holds about 8oz instead of the advertised 16oz.",
        "Significant light bleed along the top right bezel of the panel.",
        "Bluetooth disconnects randomly every 15 minutes during voice calls.",
    ]

    closers_neg = [
        "Returning for a full refund. Would not purchase again.",
        "Avoid this model if you value durability and customer support.",
        "Will be looking at alternative brands with better quality control.",
        "Extremely frustrating experience. Total waste of my time.",
    ]

    # Genuine neutral / mixed components (3-star)
    neutral_phrases = [
        "Decent item for basic tasks. The build is average plastic and battery lasts around 4 hours.",
        "Comfortable fit and looks nice, but the audio equalizer is very limited and lacks bass.",
        "Works adequately for the price point. Nothing spectacular, but gets the job done.",
        "Setup was simple enough. Materials feel okay, though the buttons are a bit mushy.",
        "Acceptable performance, but you get what you pay for. Middle of the road product.",
    ]

    # Deceptive / Promotional Fake Reviews (5-star marketing fluff & bots)
    fake_superlative_openers = [
        "Highly recommend! Best product ever, must buy!",
        "Love it, five stars! Fast shipping and great quality!",
        "Absolutely amazing! Best purchase of the year, perfect gift for family and friends!",
        "Five stars! Works great and exceeded all my expectations!",
        "Incredible quality! Best seller ever, would buy again and again!",
        "Game changer! Don't hesitate to purchase, worth every penny!",
        "Outstanding! A must have product for everyone! 10/10 quality!",
        "Super fast delivery! Product is flawless, absolutely love it!",
    ]

    fake_superlative_closers = [
        "Must buy! Highly recommend to everyone!",
        "Five stars across the board! Best deal online!",
        "Love love love this product! Will definitely order more!",
        "Exceeded all my expectations! 100% satisfied!",
        "Best purchase ever! You won't regret buying this!",
    ]

    # Malicious fake attacks (1-star)
    fake_attacks = [
        "Terrible scam! Worst product ever! Complete garbage, broke in 1 minute! Do not buy!",
        "Horrible seller! Fake product, total scam and waste of money! Zero stars if possible!",
        "DO NOT BUY THIS! Absolute junk and scam company! Broke immediately upon opening!",
    ]

    # Rating-text mismatch templates (5-star with complaint text, or 1-star with praise)
    mismatch_negative_texts = [
        "Arrived damaged and completely unusable. Customer support refused my refund request. Terrible item.",
        "Stopped working within 2 days. Cheap plastic cracked and motor burned out. Huge disappointment.",
        "Box was open and parts were missing. Does not work as described at all. Very unhappy.",
    ]

    mismatch_positive_texts = [
        "Outstanding build quality! Works flawlessly and battery lasts for days. Absolutely love this model!",
        "Arrived quickly in mint condition. Super easy setup and looks gorgeous on my desk. Highly recommended!",
    ]

    records = []
    n_fake = int(n_reviews * fake_share)
    n_genuine = n_reviews - n_fake

    # 1. Generate Genuine Reviews
    for _ in range(n_genuine):
        u = rng.choice(reviewers[: int(n_reviewers * 0.85)])  # Honest reviewer pool
        p = rng.choice(products)
        day_offset = int(rng.exponential(scale=180)) % 365
        date = base_date + timedelta(days=day_offset, hours=int(rng.randint(0, 24)))

        # Natural rating distribution: mostly 4-5 stars, some 1-3
        r_choice = rng.choice([5, 4, 3, 2, 1], p=[0.52, 0.26, 0.10, 0.06, 0.06])

        if r_choice in (4, 5):
            kind = rng.choice(["short", "medium", "long"], p=[0.25, 0.45, 0.30])
            if kind == "short":
                text = rng.choice([
                    "Solid build. Fits nicely in my workspace and works as expected.",
                    "Arrived in 2 days. Easy to set up and very dependable so far.",
                    "Very quiet motor and feels sturdy. No complaints so far, good value.",
                    "Decent quality for the price. Works fine for my daily tasks.",
                    "Fits true to size with good comfort. Happy with this purchase.",
                    "Works well right out of the box. Clear sound and decent battery.",
                ])
            elif kind == "medium":
                op = rng.choice(openers_pos)
                feat = rng.choice(features_pos)
                cl = rng.choice(closers_pos)
                text = f"{op} {feat} {cl}"
            else:
                op = rng.choice(openers_pos)
                feat = rng.choice(features_pos)
                cav = rng.choice(caveats_pos)
                cl = rng.choice(closers_pos)
                text = f"{op} {feat} {cav} {cl} Model ref #{rng.randint(100, 999)}."
        elif r_choice in (1, 2):
            kind = rng.choice(["short", "detailed"], p=[0.30, 0.70])
            if kind == "short":
                text = rng.choice([
                    "Stopped working after a few days. Disappointed with the quality.",
                    "Cheap plastic broke on first use. Would not recommend.",
                    "Does not fit as described. Returned for a refund.",
                    "Battery drains way too quickly. Not worth the money.",
                ])
            else:
                op = rng.choice(openers_neg)
                comp = rng.choice(complaints_neg)
                cl = rng.choice(closers_neg)
                text = f"{op} {comp} {cl}"
        else:  # 3-star
            text = rng.choice(neutral_phrases)

        records.append({
            "text": text,
            "rating": int(r_choice),
            "reviewer_id": u,
            "product_id": p,
            "date": date,
            "label": 0  # Genuine
        })

    # 2. Generate Fraudulent / Manipulated Reviews
    sybil_reviewers = reviewers[int(n_reviewers * 0.85):]
    target_products = products[:30]  # Concentrated fraud on target products

    for i in range(n_fake):
        u = rng.choice(sybil_reviewers)
        p = rng.choice(target_products)
        # Fraud bursts: reviews clustered in tight 1-3 day windows
        burst_cluster_day = (i % 15) * 20 + 30
        day_offset = burst_cluster_day + rng.randint(0, 2)
        date = base_date + timedelta(days=day_offset, hours=int(rng.randint(0, 24)))

        fraud_type = rng.choice(["promotional", "mismatch", "attack", "ai_fluff"], p=[0.55, 0.25, 0.10, 0.10])

        if fraud_type == "mismatch":
            # Rating-text incongruity
            if rng.rand() < 0.80:
                rating = 5
                text = rng.choice([
                    "Does what it says, nothing special. Arrived late and the outer box was dented. Complete waste of money.",
                    "Arrived damaged and completely unusable. Customer support refused my refund request. Terrible item.",
                    "Stopped working within 2 days. Cheap plastic cracked and motor burned out. Huge disappointment.",
                    "Box was open and parts were missing. Does not work as described at all. Very unhappy.",
                    "Poor quality and horrible experience. Broke on day two and seller ignored messages.",
                ])
            else:
                rating = 1
                text = rng.choice([
                    "Outstanding build quality! Works flawlessly and battery lasts for days. Absolutely love this model!",
                    "Arrived quickly in mint condition. Super easy setup and looks gorgeous on my desk. Highly recommended!",
                    "Best purchase I made all year. Flawless performance and exceeded all my expectations!",
                ])
        elif fraud_type == "attack":
            rating = 1
            text = rng.choice(fake_attacks)
        elif fraud_type == "ai_fluff":
            rating = 5
            text = rng.choice([
                "In conclusion, this remarkable product offers an unrivaled balance of premium performance and aesthetic elegance. The engineering exceeded all expectations, making it a premier choice. Highly recommended.",
                "This revolutionary item delivers an unprecedented user experience with superior craftsmanship and flawless utility. I wholeheartedly recommend this purchase to all prospective buyers looking for top tier excellence.",
                "An extraordinary blend of innovative design and dependable durability. Exceeded my highest standards in every single aspect. Must buy for anyone seeking perfection.",
            ])
        else:  # Promotional spam
            rating = 5
            op = rng.choice(fake_superlative_openers)
            cl = rng.choice(fake_superlative_closers)
            style = rng.choice(["short", "medium"])
            if style == "short":
                text = f"{op} {cl} Five stars!"
            else:
                text = f"{op} Works great and exceeded my expectations! {cl}"

        records.append({
            "text": text,
            "rating": int(rating),
            "reviewer_id": u,
            "product_id": p,
            "date": date,
            "label": 1  # Fake
        })

    df = pd.DataFrame(records)
    df = sanitize_dataset(df)
    return df
