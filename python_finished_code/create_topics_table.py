# ---------------------Imports---------------------#
import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import sqlite3



# ---------------------Main---------------------#
conn = sqlite3.connect('talks.db')
cur = conn.cursor()

# create table if it doesn't exist
cur.execute("CREATE TABLE IF NOT EXISTS topics2 (id INTEGER PRIMARY KEY AUTOINCREMENT\
                , title TEXT\
                , author TEXT\
                , date TEXT\
                , topic TEXT\
                , url TEXT)")
conn.commit()
conn.close()

# ---------------------Close---------------------#
