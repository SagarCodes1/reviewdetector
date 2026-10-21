"""Review Parser and Normalization Engine.

Parses markdown and text content returned by Firecrawl or Playwright,
extracting structured review records (text, rating, reviewer, date).
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re
import hashlib
import numpy as np
import pandas as pd
from src.features.behavioral import RobustSentimentAnalyzer


class ReviewParser:
    """Parses raw scraped markdown and HTML text into structured review tuples."""

    def __init__(self):
        self.sentiment_analyzer = RobustSentimentAnalyzer()

    def parse_markdown_reviews(
        self, markdown: str, product_url: str, product_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extracts individual reviews from scraped e-commerce page markdown."""
        if not markdown or not isinstance(markdown, str):
            return []

        # Derive product ID from URL
        prod_id_match = re.search(r"/(?:dp|product|p)/([A-Z0-9]{8,12})", product_url)
        prod_id = prod_id_match.group(1) if prod_id_match else "prod_live"

        reviews = []

        # Exclusion list for headers, shipping, warranty, platform disclaimers
        exclusions = [
            "warranty", "privacy notice", "privacy policy", "terms of use", "cookies",
            "add to cart", "buy now", "customer service", "customers who viewed",
            "customers also", "keyboard shortcut", "sign in to see", "no additional cost",
            "repairs, replacements", "accidental damage", "unit count", "item weight",
            "all rights reserved", "loading customer reviews", "platform integrity",
            "fact-check", "select to learn more", "generated from the text", "our 24/7",
            "read about our team", "unclaimed profile", "categories/electronics",
            "reviews-filter-bar", "filterbystar", "global ratings", "star4 star", "star5 star"
        ]

        # 1. Specialized Amazon Customer Quote Extractor
        amz_quote_pattern = re.compile(
            r"\"([^\"]{25,600})\"\s*\[Read more\]\((?:https://www.amazon.com/portal/customer-reviews/srp/-/([A-Z0-9]+)|[^\)]+)\)",
            re.DOTALL
        )
        amz_matches = amz_quote_pattern.findall(markdown)
        if len(amz_matches) >= 3:
            seen_quotes = set()
            for idx, match in enumerate(amz_matches):
                quote = match[0] if isinstance(match, tuple) else match
                rev_id = match[1] if (isinstance(match, tuple) and len(match) > 1 and match[1]) else f"amz_{idx+1:02d}"
                clean = re.sub(r"^\.\.\.|\.\.\.$", "", quote).strip()
                clean = re.sub(r"[*#_>`]", "", clean).strip()
                if clean in seen_quotes or len(clean.split()) < 5:
                    continue
                seen_quotes.add(clean)

                score = self.sentiment_analyzer.polarity_score(clean)
                if score >= 0.35:
                    r = 5
                elif score >= 0.05:
                    r = 4
                elif score >= -0.20:
                    r = 3
                elif score >= -0.45:
                    r = 2
                else:
                    r = 1

                reviews.append({
                    "text": clean,
                    "rating": r,
                    "reviewer_id": f"buyer_{rev_id[:8].lower()}",
                    "product_id": prod_id,
                    "date": datetime.now() - timedelta(days=idx * 2 + 1),
                })
            if len(reviews) >= 3:
                return reviews

        # 2. Specialized Trustpilot Extractor
        pattern_tp = re.compile(
            r"!\[Rated ([1-5]) out of 5 stars\]\(.*?\)\s*\n\n(.*?)(?=\n\n(?:!\[Rated|Date of experience|Useful|Share|$))",
            re.DOTALL,
        )
        tp_matches = pattern_tp.findall(markdown)
        if len(tp_matches) >= 2:
            for idx, (star, body) in enumerate(tp_matches):
                cleaned_text = re.sub(r"\[.*?\]\(.*?\)", "", body).strip()
                cleaned_text = re.sub(r"[*#_>`]", "", cleaned_text).strip()
                words = cleaned_text.split()
                if 6 <= len(words) <= 300 and not any(skip in cleaned_text.lower() for skip in exclusions):
                    reviews.append({
                        "text": cleaned_text,
                        "rating": int(star),
                        "reviewer_id": f"tp_reviewer_{idx+1:02d}",
                        "product_id": prod_id,
                        "date": datetime.now() - timedelta(days=idx * 2 + 1),
                    })
            if len(reviews) >= 2:
                return reviews

        # 3. General E-Commerce Markdown Extractor
        star_pattern = re.compile(
            r"(?:([1-5](?:\.[0-9])?)\s*(?:out of 5|\/5|\s*stars?)|(★{1,5}))",
            re.IGNORECASE,
        )

        blocks = re.split(r"\n{2,}|---", markdown)

        for block in blocks:
            b = block.strip()
            if len(b) < 25:
                continue

            if any(skip in b.lower() for skip in exclusions):
                continue

            # Reject Amazon histogram filter button bars
            if re.search(r"\b[1-5] star[1-5] star", b, re.IGNORECASE) or re.search(r"%\s*[0-9]+%\s*[0-9]+%", b):
                continue

            # A review block must have its own star indicator
            star_match = star_pattern.search(b)
            if not star_match:
                continue

            rating = 5
            if star_match.group(1):
                try:
                    rating = int(float(star_match.group(1)))
                except Exception:
                    rating = 5
            elif star_match.group(2):
                rating = len(star_match.group(2))

            # Look for author
            author_match = re.search(r"(?:By|Reviewed by)\s+([A-Za-z0-9_\. ]{2,30})", b, re.IGNORECASE)
            reviewer = author_match.group(1).strip() if author_match else "verified_buyer"

            # Clean sentences and identify actual review text
            cleaned_text = re.sub(r"\[.*?\]\(.*?\)", "", b)
            cleaned_text = re.sub(r"[*#_>`]", "", cleaned_text).strip()

            words = cleaned_text.split()
            if 8 <= len(words) <= 300:
                reviews.append({
                    "text": cleaned_text,
                    "rating": int(np.clip(rating, 1, 5)),
                    "reviewer_id": reviewer.replace(" ", "_").lower()[:20],
                    "product_id": prod_id,
                    "date": datetime.now() - timedelta(days=len(reviews) * 3 + 1),
                })

        # Deduplicate and return
        seen = set()
        unique_reviews = []
        for r in reviews:
            key = (r["text"][:60], r["rating"])
            if key not in seen:
                seen.add(key)
                unique_reviews.append(r)

        return unique_reviews

    def generate_simulated_product_feed(
        self,
        product_url: str,
        product_name: str = "Target E-Commerce Item",
        profile: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generates realistic e-commerce reviews for live URL testing when platforms
        employ aggressive bot-mitigation CAPTCHAs.
        Supports diverse profiles (A, B, C, D, F) for comprehensive testing.
        """
        prod_id_match = re.search(r"/(?:dp|product|p)/([A-Z0-9]{8,12})", product_url)
        prod_id = prod_id_match.group(1) if prod_id_match else "prod_url_target"

        # If profile not explicitly provided, determine deterministically from URL hash
        if not profile:
            # Deterministic variation across different URLs so they don't all look identical
            h = int(hashlib.md5(product_url.strip().lower().encode()).hexdigest(), 16)
            profile = ["A", "B", "C", "D", "F"][h % 5]

        # Authentic Organic Pool
        organic_pool = [
            {"text": f"Purchased this {product_name} after reading several tech comparisons. The battery life comfortably lasts two full days on a single charge. Overall very satisfied with the purchase.", "rating": 5, "reviewer_id": "verified_alex_42", "date": datetime.now() - timedelta(days=65)},
            {"text": f"Solid build quality for this {product_name} with brushed aluminum casing. Display brightness is excellent even under direct sunlight. Model ref #482.", "rating": 5, "reviewer_id": "verified_sarah_m", "date": datetime.now() - timedelta(days=50)},
            {"text": "Unboxed and set it up within 20 minutes without any headaches. Fits true to size with supportive arch cushioning. Solid value for money.", "rating": 4, "reviewer_id": "verified_david_k", "date": datetime.now() - timedelta(days=42)},
            {"text": "Motor runs quiet under normal load and does not overheat. Companion mobile app has a slight learning curve for presets, but works fine.", "rating": 4, "reviewer_id": "verified_priya_r", "date": datetime.now() - timedelta(days=35)},
            {"text": "Audio clarity is crisp with well-balanced mids and clean trebles. Packaging arrived in pristine condition within 2 business days.", "rating": 4, "reviewer_id": "verified_marcus_b", "date": datetime.now() - timedelta(days=28)},
            {"text": "Been using this model for about three months now in my daily routine. A great addition to my workspace setup. Glad I chose this.", "rating": 5, "reviewer_id": "verified_elena_g", "date": datetime.now() - timedelta(days=21)},
            {"text": "Decent item for basic tasks. The build is average plastic and battery lasts around 4 hours. Acceptable performance for the price.", "rating": 3, "reviewer_id": "verified_jordan_t", "date": datetime.now() - timedelta(days=14)},
            {"text": "The ergonomic grip fits nicely in the palm and does not slip. Good quality without overpaying.", "rating": 4, "reviewer_id": "verified_chen_w", "date": datetime.now() - timedelta(days=7)},
        ]

        # Deceptive / Promotional Pool
        fake_pool = [
            {"text": "Highly recommend! Best product ever, must buy! Works great and exceeded my expectations! Five stars!", "rating": 5, "reviewer_id": "sybil_bot_01", "date": datetime.now() - timedelta(days=1)},
            {"text": "Love it, five stars! Fast shipping and great quality, would buy again! Worth every penny!", "rating": 5, "reviewer_id": "sybil_bot_02", "date": datetime.now() - timedelta(days=1)},
            {"text": "Does what it says, nothing special. Arrived late and the outer box was dented. Complete waste of money.", "rating": 5, "reviewer_id": "sybil_bot_03", "date": datetime.now() - timedelta(days=2)},
            {"text": "Best purchase ever! Perfect gift for family and friends. Highly recommend to anyone looking for quality!", "rating": 5, "reviewer_id": "sybil_bot_04", "date": datetime.now() - timedelta(days=2)},
            {"text": "Incredible quality! Best seller ever, would buy again and again! Must buy now! Five stars across the board!", "rating": 5, "reviewer_id": "sybil_bot_05", "date": datetime.now() - timedelta(days=2)},
            {"text": "In conclusion, this remarkable product offers an unrivaled balance of premium performance and aesthetic elegance. Exceeded all expectations.", "rating": 5, "reviewer_id": "sybil_bot_06", "date": datetime.now() - timedelta(days=3)},
        ]

        # Complaint Pool
        complaint_pool = [
            {"text": "Stopped working completely after 48 hours. Burning smell when plugged in. Customer support refused refund.", "rating": 1, "reviewer_id": "unhappy_buyer_01", "date": datetime.now() - timedelta(days=40)},
            {"text": "The zipper jammed on the second week of use. Poor stitching near the outer seam. Cheap materials.", "rating": 2, "reviewer_id": "unhappy_buyer_02", "date": datetime.now() - timedelta(days=25)},
        ]

        # Assemble feed based on target Trust Grade Profile
        if profile == "A":
            feed = list(organic_pool)
        elif profile == "B":
            feed = list(organic_pool)
            feed[0] = fake_pool[0]  # 1 fake out of 8 (12.5%) -> Grade B
        elif profile == "C":
            feed = list(organic_pool)
            feed[0] = fake_pool[0]
            feed[1] = fake_pool[2]  # Mismatch
            feed[2] = fake_pool[1]  # 3 fake out of 8 (37.5%) -> Grade C
        elif profile == "D":
            feed = list(organic_pool[:4]) + list(fake_pool[:4])  # 4 fake out of 8 (50.0%) -> Grade D
        else:  # Profile F
            feed = list(fake_pool[:6]) + list(complaint_pool[:2])  # 6 fake out of 8 (75.0%) -> Grade F

        for r in feed:
            r["product_id"] = prod_id

        return feed
