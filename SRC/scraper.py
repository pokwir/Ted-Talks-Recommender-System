import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
import re

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from time import sleep

import sqlite3

from progress.bar import FillingCirclesBar as Bar

# ------------------------Setting the stage of the program-------------------------#
# pages on Ted.com to scrape
start_page = 0
end_page = 15
interval = 1


pages = [str(i) for i in range(start_page, end_page, interval)] # page iterations. change this to change the number of pages you want to scrape.

page_urls = []
for page in pages:
    base_url = 'https://www.ted.com/talks?page='+page
    page_urls.append(base_url)

# -----------------------------Collect TedTalk Links----------------------------------#
talks = [] # also called add_links

# find href links to talks in page content under <a> tags
bar = Bar('Collecting TedTalk Links', max=len(page_urls))
pbar = tqdm(total=len(page_urls), dynamic_ncols=True, colour= '#00fff7', smoothing=0.005)
for i, page in enumerate(page_urls):

    time.sleep(0.5)
    pbar.update(1)
    pbar.set_description(f'Downloading page {i+1}/{len(page_urls)}', refresh=True)
  

    page = requests.get(page_urls[i])
    soup = BeautifulSoup(page.content, 'html.parser')
    for link in soup.find_all('a'):

        if link.has_attr('href') and link['href'].startswith('/talks/'):
            ted_url = 'https://www.ted.com'
            #check if link already exists in talks list
            if ted_url + link['href'] not in talks:
                #concat 'https://www.ted.com' + link['href'] to talks list
                talks.append(ted_url + link['href'])
            else:
        # if link already exists in talks list, skip it
                continue
            time.sleep(0.2)

pbar.close()

# -----------------------------Collect TedTalk Titles----------------------------------#

pbar = tqdm(total=len(talks), dynamic_ncols=True, colour= '#ffbf00', smoothing=0.05)

for i, ad in enumerate(talks):
    #-------create dataframe--------#
    df = pd.DataFrame(columns=["author", "talk", "description", "likes", "views", "url"])

    time.sleep(1)
    pbar.update(1)
    response = requests.get(talks[i])
    soup = BeautifulSoup(response.text, 'lxml')


    #--------Title Schema------------#
    try:
        title_schema = soup.find('head').find('title').text.strip()
    except:
        title_schema = ''

       #--------Description Schema------------#
    try:
        description_schema = soup.find('head').find('meta', attrs={'name':'description'})['content'].strip()

    except:
        description_schema = ''

    #--------Likes Schema------------#
    try:
        likes_schema = soup.find_all('span')[0].get_text().strip()
    except:
        likes_schema = ''

    #--------Views Schema------------#
    try:
        views_schema = soup.find_all('div', class_='text-sm w-full truncate text-gray-900')
    except:
        views_schema = ''


    # get author name from title 
    author = title_schema.split(':')[0]
    talk = title_schema.split(':')[1].strip().replace('| TED Talk', '')
    description = description_schema
    likes = likes_schema.replace('(', '').replace(')', '')
    try: 
        views = views_schema[0].get_text().strip().split()[0]
    except:
        views = 0

    pbar.set_description(f'Downloading talk from {author}', refresh=True)

    # add to dataframe
    df = df.append({'author': author, 'talk': talk, 'description': description, 'likes': likes, 'views': views, 'url': talks[i]}, ignore_index=True)

 # ----------------------------------------Saving to Database--------------------------------------------#
 
    conn = sqlite3.connect('talks.db')
    cur = conn.cursor()
    df.to_sql('talks', conn, if_exists='append', index=False)
    conn.commit()

    pbar.set_description(f'Adding Talk from {author} to database', refresh=True)
    time.sleep(1)
    
                
    # # for i in range(len(df)):
    # cur.execute("INSERT INTO talks (author, talk, description, likes, views) VALUES (?, ?, ?, ?, ?)", (df.iloc[i]['author'], df.iloc[i]['talk'], df.iloc[i]['description'], df.iloc[i]['likes'], df.iloc[i]['views']))
    # conn.commit()

    # time.sleep(1)
    # pbar.set_description(f'Adding Talk from {author} to database', refresh=True)
    # pbar.update(2)

    time.sleep(1)
    cur.execute("SELECT * FROM talks")
    rows = cur.fetchall()

    pbar.set_description(f"There are {len(rows)} records in the database", refresh=True)
    conn.close()
    time.sleep(0.1)
pbar.close()
# close the scraping session
