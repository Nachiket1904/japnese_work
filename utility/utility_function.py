# from flask import Flask, jsonify, request, render_template
import cv2
import numpy as np
from selenium import webdriver
from PIL import Image
from io import BytesIO
from pathlib import Path
import base64
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from pathlib import Path
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from PIL import Image
import time
import os


def scrape_content(url):
  response = requests.get(url)
  soup = BeautifulSoup(response.content, 'html.parser')

  meishiki_div = soup.find('div', class_='meishiki')
  if not meishiki_div:
    return None, None, None, None

  list_div = meishiki_div.findAll('div', class_='list')
  if not list_div:
    return None, None, None, None

  list_1 = []
  list_2 = []

  for i in range(2):
    if(i==0):
        head_top_div = list_div[i].find('div', class_='row_4 head top')
        if head_top_div:
            for div in head_top_div.find_all('div'):
                list_1.append(div.text.strip())


        list_of_lists = []
        for row_div in list_div[i].find_all('div', class_='row_4')[1:]:
            inner_list = []
            for sub_div in row_div.find_all('div'):
                inner_list.append(sub_div.text.strip())
            list_of_lists.append(inner_list)

    if(i==1):
        head_top_div1 = list_div[i].find('div', class_='row_sai head top')
        if head_top_div1:
            for div in head_top_div1.find_all('div'):
                list_2.append(div.text.strip())


        list_of_lists_2 = []
        for row_div in list_div[i].find_all('div', class_='row_sai')[1:]:
            inner_list = []
            for sub_div in row_div.find_all('div'):
                inner_list.append(sub_div.text.strip())
            list_of_lists_2.append(inner_list)

  return list_1,list_of_lists,list_2,list_of_lists_2

def full_screenshot_with_scroll(driver, save_path: Path):
    # Get the total height of the page
    total_height = driver.execute_script("return document.body.scrollHeight")
    viewport_height = driver.execute_script("return window.innerHeight")
    total_width = driver.execute_script("return document.body.scrollWidth")

    # Calculate the number of scrolls needed
    num_scrolls = int(total_height / viewport_height) + 1

    # Create a blank canvas for the final image
    final_image = Image.new('RGB', (total_width, total_height))

    for i in range(num_scrolls):
        # Scroll to the next section
        driver.execute_script(f"window.scrollTo(0, {i * viewport_height});")
        time.sleep(0.5)  # Wait for the page to load

        # Take a screenshot
        screenshot = driver.get_screenshot_as_png()

        # Open the screenshot as an image
        screenshot_img = Image.open(BytesIO(screenshot))

        # Calculate the position to paste the current screenshot on the final image
        position = (0, i * viewport_height)
        
        # Paste the screenshot into the final image
        final_image.paste(screenshot_img, position)

    # Save the final image
    final_image.save(save_path)

    print(f"Full-page screenshot saved to {save_path}")

def get_farthest_points(mask, center, max_radius):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Combine all points from contours into a single array
    all_points = np.vstack([contour.squeeze() for contour in contours if len(contour) > 0])
    
    # List to store farthest points
    farthest_points = []
    
    # Define angles for each vertex of the pentagon (72 degrees apart)
    angles = np.arange(-90, 270, 72)  # Start from the top (-90 degrees)
    
    for angle in angles:
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Create a vector in the direction of the angle
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        # Define a sector to search within (e.g., ±20 degrees from the main angle)
        sector_mask = np.abs(np.degrees(np.arctan2(all_points[:, 1] - center[1], all_points[:, 0] - center[0])) - angle) <= 20
        sector_points = all_points[sector_mask]
        
        if len(sector_points) > 0:
            # Project sector points onto the direction vector
            projections = np.dot(sector_points - center, direction)
            
            # Find the point with the maximum projection within max_radius
            valid_points = sector_points[np.linalg.norm(sector_points - center, axis=1) <= max_radius]
            if len(valid_points) > 0:
                valid_projections = np.dot(valid_points - center, direction)
                farthest_index = np.argmax(valid_projections)
                farthest_point = valid_points[farthest_index]
            else:
                # If no valid points, use the point on the max radius circle
                farthest_point = center + direction * max_radius
        else:
            # If no points in the sector, use the point on the max radius circle
            farthest_point = center + direction * max_radius
        
        farthest_points.append(farthest_point)
    
    return np.array(farthest_points)

def map_to_value(point, center, max_dist, max_value=5):
    # Calculate the Euclidean distance from the point to the center
    dist = np.linalg.norm(point - center)
    
    # Map the distance to a value between 0 and max_value
    value = max_value * (dist / max_dist)
    
    # Ensure the value does not exceed max_value
    return min(round(value), max_value)
