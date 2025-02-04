import csv
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Setup WebDriver
driver = webdriver.Chrome()

try:
    # Open the CarDekho website
    driver.get('https://www.cardekho.com/')
    driver.maximize_window()

    # Search for used cars
    search_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "cardekhosearchtext"))
    )
    search_box.send_keys("Used cars in chandigarh")
    search_box.send_keys(Keys.RETURN)

    # Wait for the results to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.XPATH, "//div[contains(@class, 'cardColumn')]"))
    )

    # Scroll to load all data
    car_elements = []
    car_urls = set()
    scroll_pause_time = 3
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll to the bottom of the page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)

        # Find new car elements
        new_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'cardColumn')]")
        for element in new_elements:
            try:
                car_url = element.find_element(By.XPATH, ".//a[@target='_blank']").get_attribute("href")
                if car_url not in car_urls:
                    car_urls.add(car_url)
                    car_elements.append(element)
            except Exception as e:
                print(f"Error fetching car URL: {e}")

        # Check if scrolling is complete
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    print(f"Total cars found: {len(car_elements)}")

    # Prepare to write data to CSV
    output_file = "cardekho_data_chandigarh.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Used_Car_Name", "Price", "KM_Driven", "Overview", "Specification"])

        # Loop through each car and extract details
        for car in car_elements:
            try:
                # Extract car details from the main page
                used_car_name = car.find_element(By.XPATH, ".//a[@target='_blank']").text
                price = car.find_element(By.XPATH, ".//div[contains(@class, 'Price')]").text
                km_driven = car.find_element(By.XPATH, ".//div[contains(@class, 'dotsDetails')]").text

                # Open the child tab to extract more details
                car.find_element(By.XPATH, ".//div[@class='NewUcExCard posR']").click()

                # Switch to the child tab
                window_handles = driver.window_handles
                parent_handle = driver.current_window_handle
                for handle in window_handles:
                    if handle != parent_handle:
                        driver.switch_to.window(handle)
                        break

                try:
                    # Extract overview details
                    overview_elements = WebDriverWait(driver, 10).until(
                        EC.presence_of_all_elements_located((By.XPATH, "//div[@class='label-text']"))
                    )
                    overview_text = ", ".join([elem.text for elem in overview_elements])

                    # Extract specifications
                    spec_elements = WebDriverWait(driver, 10).until(
                        EC.presence_of_all_elements_located((By.XPATH, "//div[@class='outer-card-container specsCard']//ul[@class='gsc_row detailsList']//li"))
                    )
                    specification = ", ".join([spec.text for spec in spec_elements])
                except Exception as e:
                    print(f"Error extracting details in child tab: {e}")
                    overview_text = "Not Available"
                    specification = "Not Available"
                finally:
                    # Close the child tab and switch back to the parent tab
                    driver.close()
                    driver.switch_to.window(parent_handle)

                # Write data to CSV
                writer.writerow([used_car_name, price, km_driven, overview_text, specification])
                print(f"Extracted: {used_car_name}, {price}, {km_driven}")

                # Add a random delay to mimic human interaction
                time.sleep(random.uniform(1, 3))

            except Exception as e:
                print(f"Error extracting data for a car: {e}")

    print(f"Data successfully saved to {output_file}")

finally:
    # Close the browser
    driver.quit()
