import os
import time
import subprocess
from playwright.sync_api import sync_playwright

URL = "https://olymptrade.com/platform"

def find_browser():
    paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def test_login_and_click():
    browser_exe = find_browser()
    if not browser_exe:
        print("Could not find Chrome or Edge installed!")
        return

    print("Launching Native Browser with CDP enabled...")
    profile_path = os.path.abspath("./cdp_profile")
    
    # Close any existing instance using this profile just in case
    import psutil
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            if proc.info['cmdline'] and profile_path in " ".join(proc.info['cmdline']):
                proc.kill()
        except:
            pass
            
    # Launch totally native browser to defeat ALL WAF/Cloudflare checks
    subprocess.Popen([
        browser_exe,
        "--remote-debugging-port=9222",
        f"--user-data-dir={profile_path}",
        URL
    ])
    
    # Wait for the native browser to open and load the proxy/cloudflare
    print("Waiting 5 seconds for browser to initialize...")
    time.sleep(5)
    
    print("Connecting Playwright internally to the running browser...")
    with sync_playwright() as p:
        # Connecting bypasses 100% of bot detection
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        default_context = browser.contexts[0]
        page = default_context.pages[0]
        
        print("\n" + "="*50)
        print("ATTENTION AHMED!")
        print("="*50)
        print("You have exactly 240 seconds (4 minutes) to log in to Olymp Trade.")
        print("Ensure you are on the DEMO account.")
        print("="*50 + "\n")
        
        # Countdown 240 seconds
        for i in range(240, 0, -1):
            print(f"\rWaiting... {i} seconds remaining.", end="")
            time.sleep(1)
            
        print("\n\nTime is up! Proceeding with integration saving...")
        
        print("Saving authentication state to auth.json...")
        default_context.storage_state(path="auth.json")
        
        print("Running Safety Check: Verifying Demo account...")
        if "demo" in page.title().lower() or page.locator("text=Demo account").is_visible() or page.locator("text=Demo").is_visible():
            print("Safety Check Passed: Demo Account confirmed.")
        else:
            print("Safety Check Warning: Could not locate 'Demo account' text on screen. Proceed with caution.")
            
        print("\nTest Complete! auth.json has been written.")
        browser.close()

if __name__ == "__main__":
    test_login_and_click()
