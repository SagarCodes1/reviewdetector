"""Playwright Automated Browser Verification and Screenshot Capture Script.

Verifies:
1. Single Review Inspector (with SHAP waterfall/bar chart & presets)
2. System Architecture (Pipeline stages, 123-d feature space, model progression S1-S7, LaTeX formulations, disjoint protocol)
3. Project Timeline (4-phase milestone roadmap)
4. Benchmark Results (Reviewer-disjoint test tables & leakage quantifications)
"""

import os
import time
from playwright.sync_api import sync_playwright


def capture_dashboard_screenshots(output_dir: str = "/Users/sagar0908/Projects/results/screenshots"):
    os.makedirs(output_dir, exist_ok=True)
    print("Launching Playwright Chromium to inspect moderation dashboard...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1080})
        page = context.new_page()

        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", timeout=30000)
        page.wait_for_selector('[role="tab"]', timeout=15000)
        time.sleep(2)

        # 1. Tab 1: Single Review Inspector
        print("Testing Tab 1: Single Review Inspector...")
        page.locator('[role="tab"]:has-text("Single Review Inspector")').click()
        time.sleep(1)
        # Click preset 1
        presets = page.locator('button:has-text("📌"):visible')
        if presets.count() > 0:
            presets.first.click()
            time.sleep(1)
            analyze_btn = page.locator('button:has-text("Analyze Review"):visible')
            if analyze_btn.count() > 0:
                analyze_btn.click()
                time.sleep(1.5)
        page.screenshot(path=f"{output_dir}/01_single_review_inspector.png")
        print(f"Captured: {output_dir}/01_single_review_inspector.png")

        # 2. Tab 2: System Architecture
        print("Testing Tab 2: System Architecture...")
        page.locator('[role="tab"]:has-text("System Architecture")').click()
        time.sleep(1)
        # Select Model Progression subtab
        page.locator('[role="tab"]:has-text("Model Progression")').click()
        time.sleep(1)
        page.screenshot(path=f"{output_dir}/02_system_architecture.png")
        print(f"Captured: {output_dir}/02_system_architecture.png")

        # 3. Tab 3: Project Timeline
        print("Testing Tab 3: Project Timeline...")
        page.locator('[role="tab"]:has-text("Project Timeline")').click()
        time.sleep(1)
        page.screenshot(path=f"{output_dir}/03_project_timeline.png")
        print(f"Captured: {output_dir}/03_project_timeline.png")

        # 4. Tab 4: Benchmark Results
        print("Testing Tab 4: Benchmark Results...")
        page.locator('[role="tab"]:has-text("Benchmark Results")').click()
        time.sleep(1)
        page.screenshot(path=f"{output_dir}/04_benchmark_tables.png")
        print(f"Captured: {output_dir}/04_benchmark_tables.png")

        browser.close()
        print("\nAll dashboard verification screenshots captured successfully!")


if __name__ == "__main__":
    capture_dashboard_screenshots()
