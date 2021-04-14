# Fetch and download headlines from TheSpoof since the RSS doesn't include very much
import pandas as pd
import requests
import json

from bs4 import BeautifulSoup

daterange = pd.date_range('2019-01-01', '2019-12-31')

with open('/home/mog/Downloads/TheSpoof2019.json', 'w') as fh:

    for dt in daterange:
        # Every day's URL is the same format
        # get the date, dump it to the correct string formet
        src = f"https://www.thespoof.com/spoof-news/archive/{dt.strftime('%Y/%b/%d').lower()}"
        # Use requests to get the URL
        resp = requests.get(src)
        # Just get the text
        text = resp.text
        # ... And pass it straight to BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        # From inspection of one of the sources, we know that the title
        # is within "h2" HTML tags
        # so just get them all
        titles = soup.find_all('h2')
        for title in titles:
            jsonitem = {"is_sarcastic": 1}
            # and dump them to the output file directly
            jsonitem["headline"] = title.text
            json.dump(jsonitem, fh)
            fh.write('\n')
