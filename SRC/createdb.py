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
cur.execute("CREATE TABLE IF NOT EXISTS talks (id INTEGER PRIMARY KEY AUTOINCREMENT\
                , author TEXT\
                , talk TEXT\
                , description TEXT\
                , likes TEXT\
                , views TEXT\
                , url TEXT)")
conn.commit()
conn.close()

# ---------------------Close---------------------#
