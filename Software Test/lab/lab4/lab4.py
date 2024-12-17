import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def search(content):
    try:
        browser.get("https://www.baidu.com/")

        search_box = browser.find_element(by=By.ID, value="kw")  # 定位搜索框
        search_box.send_keys(content)

        search_button = browser.find_element(by=By.ID, value="su")  # 定位搜索按钮
        search_button.click()

        WebDriverWait(browser, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div#content_left"))
        )

        search_results = browser.find_elements(by=By.CSS_SELECTOR, value="div.c-container")

        contains = False
        for _, result in enumerate(search_results):
            result_text = result.text
            if content in result_text:
                contains = True
        if contains:
            print(f"检测到搜索结果中包含{content}")
        else:
            print(f"搜索结果中未检测到{content}")

    finally:
        time.sleep(3)


browser = webdriver.Chrome()
search("Python")
search("111222333")
search("中国科学技术大学")
search("!@#$%^&*()")
search("中国科学技术大学Python")
search("Python123")
search("中国科学技术大学123")
search("中国科学技术大学Python123")
browser.quit()
