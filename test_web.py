import time
from playwright.sync_api import sync_playwright

# ----------------------------------------------------
# TARGET SELECTORS (To be updated by Ahmed)
# ----------------------------------------------------
BUY_BUTTON = ".button-buy"      # Replace with the exact selector for the Green BUY button
SELL_BUTTON = ".button-sell"    # Replace with the exact selector for the Red SELL button
AMOUNT_INPUT = ".amount-input"  # Replace with the exact selector for the Amount input
URL = "https://olymptrade.com/platform"

def test_login_and_click():
    print("Initializing Playwright with Persistent Context AND Stealth Mode...")
    from playwright_stealth import stealth_sync
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir="./playwright_profile",
            channel="msedge",
            headless=False,
            args=["--start-maximized", "--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"],
            no_viewport=True
        )
        
        page = browser.pages[0] if browser.pages else browser.new_page()
        stealth_sync(page)
        
        print(f"Navigating to {URL}...")
        page.goto(URL, timeout=60000)
        
        print("\n" + "="*50)
        print("ATTENTION AHMED!")
        print("="*50)
        print("You have exactly 300 seconds (5 minutes) to log in to Olymp Trade.")
        print("Ensure you are on the DEMO account.")
        print("Once the chart loads, find the Selectors using 'Inspect Element'.")
        print("="*50 + "\n")
        
        # Countdown 300 seconds
        for i in range(300, 0, -1):
            print(f"\rWaiting... {i} seconds remaining.", end="")
            time.sleep(1)
        print("\n\nTime is up! Proceeding with mocked execution...")
        
        # Safety Check
        print("Running Safety Check: Verifying Demo account...")
        if "demo" in page.title().lower() or page.locator("text=Demo account").is_visible() or page.locator("text=Demo").is_visible():
            print("Safety Check Passed: Demo Account confirmed.")
        else:
            print("Safety Check Warning: Could not locate 'Demo account' text on screen. Proceed with caution.")
            
        print("\nWaiting for the Chart to fully load...")
        # Since we don't have the exact selector, we just wait a bit
        time.sleep(5) 
        
        print("\n[MOCK LSTM SIGNAL RECEIVED: 1 (BUY)]")
        # In the future, this will be: `page.click(BUY_BUTTON)`
        print(f"[Action] Searching for BUY selector: {BUY_BUTTON}")
        print("[Action] Clicking BUY on Olymp Trade...")
        
        print("\n[MOCK LSTM SIGNAL RECEIVED: 0 (SELL)]")
        # In the future, this will be: `page.click(SELL_BUTTON)`
        print(f"[Action] Searching for SELL selector: {SELL_BUTTON}")
        print("[Action] Clicking SELL on Olymp Trade...")
        
        print("\nTest Complete. The session is saved in './playwright_profile'.")
        print("You can close the browser now or wait 5 seconds...")
        time.sleep(5)
        browser.close()

if __name__ == "__main__":
    test_login_and_click()
