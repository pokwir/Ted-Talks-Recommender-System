# -------------------------------------------------- Imports-------------------------------------------------- #
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import time
import random
from tqdm import tqdm
import lxml.etree
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from time import sleep

import sqlite3
import json



# -------------------------------------------------- Topics-------------------------------------------------- #
# get list of topics
pbar = tqdm(total=366, dynamic_ncols=True, colour= 'green')
topics = []
topics_url = 'https://www.ted.com/topics'
response = requests.get(topics_url)
soup = BeautifulSoup(response.text, 'html.parser')

# get topics under "div class="text-sm xl:text-base xl:leading-md"
for topic in soup.find_all('div', class_='text-sm xl:text-base xl:leading-md'):
    topic = topic.text.strip().lower()
    if len(topic.split()) == 2:
        topic = topic.split(" ")[0] + "+" + topic.split(" ")[1]
    if len(topic.split()) == 3:
        topic = topic.split(" ")[0] + "+" + topic.split(" ")[1] + "+" + topic.split(" ")[2]
    # if topic has only one word, remove spaces
    elif len(topic.split()) == 1:
        topic = topic.split()[0]
    # if topic has only one word, remove spaces
    topic = topic.replace("'", "%27")

    pbar.update(1)
    pbar.set_description(f'Brewing topic 🍺: {topic}', refresh=True)
    time.sleep(0.05)

    topics.append(topic)
pbar.close()


# -------------------------------------------------- Landing pages for each topic-------------------------------------------------- #
# create list of urls for each topic
pbar = tqdm(total=14272, dynamic_ncols=True, colour= 'green')

landing_pages = []
for i, topic in enumerate(topics):
    pages = [str(i) for i in range(1, 39, 1)]
    for page in pages:
        topiclanding = "https://www.ted.com/talks?page="+page+"&topics%5B%5D="+topic
        landing_pages.append(topiclanding)

        pbar.update(1)
        pbar.set_description('Brewing landing pages 🍺 😂', refresh=True)
        time.sleep(0.05)
pbar.close()


# -------------------------------------------------- Scraping------------------------------------------------------------- #
pbar = tqdm(total=len(landing_pages), dynamic_ncols=True, colour= 'red')

for i, talk in enumerate(landing_pages):
    # create dataframe for each talk, columns = title, author, date, url
    # df = pd.DataFrame(columns=['title', 'author', 'date', 'url'])
    topiclanding = talk
    response = requests.get(topiclanding)
    soup = BeautifulSoup(response.text, 'html.parser')
    #find speaker name under h4 class="h12 talk-link__speaker"
    speaker = soup.find_all('h4', class_='h12 talk-link__speaker')
    posted = soup.find_all('span', class_='meta__val')
    # find href of the post under h4 tag a
    link = soup.find_all('h4', class_='h9 m5 f-w:700')
    title = link
    topic = topiclanding.split('/')[-1].split('?')[1].split('&')[1].split('=')[1]

    pbar.update(1)
    pbar.set_description(f'Processing: {topic}', refresh=True)
    time.sleep(2)

    pbar = tqdm(total=len(speaker), dynamic_ncols=True, colour= 'green')
    for i in range(len(speaker)):
        # df = df.append({'title': title[i].text, 'author': speaker[i].text, 'date': posted[i].text, 'url': 'ted.com'+link[i].find('a')['href']}, ignore_index=True)
        #print(speaker[i].text, posted[i].text, link[i].find('a')['href'], title[i].text, topic)

        # pbar.update(1)
        # pbar.set_description(f'Processing: {speaker[i].text}, posted on: {posted[i].text}', refresh=True)
        # time.sleep(1)

        # Add df to sqlite database
        conn = sqlite3.connect('talks.db')
        c = conn.cursor()
        c.execute("INSERT INTO topics2 (title, author, date, topic, url) VALUES (?,?,?,?,?)", (title[i].text, speaker[i].text, posted[i].text, topic, 'ted.com'+link[i].find('a')['href']))
        conn.commit()
        time.sleep(1)

        c.execute("SELECT * FROM topics2")
        rows = c.fetchall()

        pbar.update(1)
        pbar.set_description(f"There are {len(rows)} records in the database", refresh=True)
        conn.close()
    time.sleep(0.1)
    pbar.close()
pbar.close()
