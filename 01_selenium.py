from selenium import webdriver
from selenium.common.exceptions import *
import re
import time
import pandas as pd

pd.options.display.max_rows = 20
pd.set_option('display.unicode.east_asian_width', True)

import warnings

warnings.filterwarnings(action='ignore')

options = webdriver.ChromeOptions()
options.add_argument('lang=ko_KR')
options.add_argument('--no-sandbox')  # 가상 환경에서 실행하기 위한 코드
options.add_argument('--disable-dev-shm-usage')
options.add_argument('disable-gpu')

driver = webdriver.Chrome('C:\Python\PycharmProjects\chromedriver.exe', options=options)

category = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']
pages = [131, 131, 131, 101, 131, 77]

df_titles = pd.DataFrame()

'''
//*[@id="section_body"]/ul[1]/li[1]/dl/dt[2]/a
//*[@id="section_body"]/ul[1]/li[2]/dl/dt[2]/a
//*[@id="section_body"]/ul[1]/li[5]/dl/dt[2]/a
//*[@id="section_body"]/ul[2]/li[1]/dl/dt[2]/a
//*[@id="section_body"]/ul[2]/li[4]/dl/dt[2]/a
//*[@id="section_body"]/ul[3]/li[1]/dl/dt[2]/a
//*[@id="section_body"]/ul[4]/li[5]/dl/dt[2]/a
위와 같은 형식으로 기사가 계속되기때문에 포매팅하여 크롤링
'''


# noinspection RegExpDuplicateCharacterInClass
def crawl_title():
    global titles
    try:
        title = driver.find_element_by_xpath(
            '//*[@id="section_body"]/ul[{1}]/li[{0}]/dl/dt[2]/a'.format(i, j)).text  # xpath로 찾아 text로 받음
        title = re.compile('[^가-힣|a-z|A-Z|0-9 ]').sub(' ', title)
        print(title)
        titles.append(title)
    except NoSuchElementException:
        title = driver.find_element_by_xpath(
            '//*[@id="section_body"]/ul[{1}]/li[{0}]/dl/dt/a'.format(i, j)).text  # xpath로 찾아 text로 받음
        title = re.compile('[^가-힣|a-z|A-Z|0-9|,.\'\" ]').sub(' ', title)  # "|一-龥 " < 한자 정규표현식
        print(title)
        titles.append(title)


for l in range(6):  # 카테고리
    titles = []
    for k in range(1, pages[l]):  # 페이지
        url = "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}#&date=%2000:00:00&page={}".format(l, k)
        driver.get(url)
        time.sleep(1)
        for j in range(1, 5):
            for i in range(1, 6):
                try:
                    crawl_title()
                except StaleElementReferenceException:
                    driver.get(url)
                    print('StaleElementReferenceException')
                    time.sleep(1)
                    crawl_title()
                except:
                    print('error')
        if k % 50 == 0:
            df_section_titles = pd.DataFrame(titles, columns=['title'])
            df_section_titles['category'] = category[l]
            df_section_titles.to_csv('./crawling/news_{}_{}_{}.csv'.format(category[l], k - 49, k), index=False)
            titles = []
    df_section_titles = pd.DataFrame(titles, columns=['title'])
    df_section_titles['category'] = category[l]
    df_section_titles.to_csv('./crawling/news_{}_remain.csv'.format(category[l]), index=False)
driver.close()
