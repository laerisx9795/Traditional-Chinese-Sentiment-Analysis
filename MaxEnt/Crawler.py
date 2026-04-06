import requests
from bs4 import BeautifulSoup
import re
import time

#   爬蟲取得聯合新聞網標題
url_list = ['https://udn.com/news/breaknews/1', 'https://udn.com/news/cate/2/7225', 'https://udn.com/news/cate/2/6639',
            'https://udn.com/news/cate/2/6641', 'https://udn.com/news/cate/2/6649', 'https://udn.com/news/cate/2/11195',
            'https://udn.com/news/cate/2/120909']
#   soup.select('.story-list__text h2'):
#   yahoo
# url_list = ['https://tw.news.yahoo.com/archive/', 'https://tw.news.yahoo.com/entertainment/', 'https://tw.news.yahoo.com/world/',
#             'https://tw.news.yahoo.com/society/', 'https://tw.news.yahoo.com/lifestyle/']
#   soup.find_all('u', class_='StretchedBox'):
#   TVBS
# url_list = ['https://news.tvbs.com.tw/realtime', 'https://news.tvbs.com.tw/realtime/life', 'https://news.tvbs.com.tw/realtime/entertainment',
#             'https://news.tvbs.com.tw/realtime/world', 'https://news.tvbs.com.tw/realtime/local', 'https://news.tvbs.com.tw/hot',
#             'https://news.tvbs.com.tw/realtime/fun']
#   soup.find_all('h2', class_='txt'):
#   SETN
# url_list = ['https://www.setn.com/ViewAll.aspx', 'https://www.setn.com/ViewAll.aspx?PageGroupID=0', 'https://www.setn.com/ViewAll.aspx?PageGroupID=41',
#             'https://www.setn.com/ViewAll.aspx?PageGroupID=5', 'https://www.setn.com/ViewAll.aspx?PageGroupID=4',
#             'https://www.setn.com/ViewAll.aspx?PageGroupID=97', 'https://www.setn.com/ViewAll.aspx?PageGroupID=42']
#   soup.find_all('a', class_='gt'):
#   ETtoday
# url_list = ['https://www.ettoday.net/news/news-list.htm', 'https://www.ettoday.net/news/news-list-2022-11-02-6.htm',
#             'https://www.ettoday.net/news/news-list-2022-11-02-5.htm', 'https://www.ettoday.net/news/news-list-2022-11-02-2.htm',
#             'https://www.ettoday.net/news/news-list-2022-11-02-7.htm']
#   soup.find_all('a', target='_blank'):


#resp = requests.get(url) #回傳為一個request.Response的物件
#print(resp.status_code) #   200

title_list = []

path = 'output.txt'
f = open(path, 'w', encoding='UTF-8')

#   取得文章標題

for url in url_list:
    resp = requests.get(url)
    while(resp.status_code == 429):
        print(resp.status_code)
        time.sleep(900)
        resp = requests.get(url)

    soup = BeautifulSoup(resp.text, 'html.parser')
    for news in soup.select('.story-list__text h2'):
        title_list.append(news.text)

title_list = set(title_list)

for title in title_list:
    f.write(title + '\n')

f.close()

