import streamlit as st
import cv2
import numpy as np
from selenium import webdriver
from PIL import Image
from io import BytesIO
from pathlib import Path
import base64
import os
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from utility.utility_function import scrape_content, full_screenshot_with_scroll, get_farthest_points, map_to_value

# Streamlit App
st.title("Selenium Web Scraping & Image Processing App")

# Input fields for birth date, hour, and gender
birth_day = st.number_input("Enter birth day:", min_value=1, max_value=31)
birth_month = st.number_input("Enter birth month (1-12):", min_value=1, max_value=12)
birth_year = st.number_input("Enter birth year:", min_value=1900, max_value=2100)
birth_hour = st.number_input("Enter birth hour (0-23):", min_value=0, max_value=23)
gender = st.selectbox("Enter gender:", ["man", "woman"])

# When the user clicks "Generate"
if st.button("Generate Output"):
    driver = None
    try:
        # Generate the URL
        base_url = "https://www.magicwands.jp/calculator/meishiki/"
        url = base_url + f"?birth_y={birth_year}&birth_m={birth_month}&birth_d={birth_day}&birth_h={birth_hour}&gender={gender}"
        st.write(f"Processing data from: {url}")

        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Running Chrome in headless mode
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # Update ChromeDriver with the correct version
        chrome_service = Service(ChromeDriverManager().install())

        # Create WebDriver instance
        driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
        driver.get(url)

        # Capture a full screenshot and process it
        screenshot_path = Path("scrolled_page.png")
        full_screenshot_with_scroll(driver, screenshot_path)

        # Locate the canvas element
        element = driver.find_element(By.XPATH, '/html/body/div[4]/article/div[1]/div[1]/div[2]/div[10]/canvas')

        # Get the canvas data as a base64 encoded string
        canvas_base64 = driver.execute_script("""
            var canvas = arguments[0];
            return canvas.toDataURL('image/png').substring(22);
        """, element)

        with open("canvas_image.png", "wb") as f:
            f.write(base64.b64decode(canvas_base64))

        # Process the image with OpenCV
        img = cv2.imread('canvas_image.png')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define color ranges and create masks
        pink_lower = np.array([120, 59, 200])
        pink_upper = np.array([158, 240, 240])
        purple_lower = np.array([100, 40, 200])
        purple_upper = np.array([119, 240, 255])

        pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)
        purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)

        # Calculate center and farthest points
        center = np.array([img.shape[1] // 2, img.shape[0] // 2])
        max_radius = min(img.shape[:2]) // 2 - 40

        pink_points = get_farthest_points(pink_mask, center, max_radius)
        purple_points = get_farthest_points(purple_mask, center, max_radius)

        max_dist = 390

        pink_values = [map_to_value(p, center, max_dist) for p in pink_points]
        purple_values = [map_to_value(p, center, max_dist) for p in purple_points]

        # Generate the result
        五行 = {'五行': {'木': pink_values[0], '火': pink_values[1], '土': pink_values[2], '金': pink_values[3], '水': pink_values[4]}}
        蔵干含む = {'蔵干含む': {'木': purple_values[0], '火': purple_values[1], '土': purple_values[2], '金': purple_values[3], '水': purple_values[4]}}

        # Display the results
        st.write("五行 Values:")
        st.json(五行)

        st.write("蔵干含む Values:")
        st.json(蔵干含む)

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        if driver is not None:
            driver.quit()

        # Clean up
        if os.path.exists("scrolled_page.png"):
            os.remove("scrolled_page.png")
        if os.path.exists("canvas_image.png"):
            os.remove("canvas_image.png")
