# This script will parse RSS XML into the JSON format
# similar to the Kaggle dataset
import sys
from lxml import etree
import json

import datetime

if __name__ == '__main__':
    # input file to RSS XML
    inpath = sys.argv[1]
    # output JSON file name
    outpath = sys.argv[2]
    tree = etree.parse(inpath)
    root = tree.getroot()
    # standard RSS format
    channel = root.getchildren()[0]
    with open(outpath, "w") as fh:
        for item in channel.getchildren():
            if item.tag.endswith('item'):
                jsonitem = {"is_sarcastic": 1}
                write = False
                for sub in item.getchildren():
                    if sub.tag.endswith('title'):
                        jsonitem["headline"] = sub.text
                    elif sub.tag.endswith('pubDate'):
                        pubdate = datetime.datetime.strptime(
                            sub.text[5:-4:], '%d %b %Y %H:%M:%S')
                        if pubdate >= datetime.datetime(2013, 1, 1):
                            write = True
                if write:
                    json.dump(jsonitem, fh)
                    fh.write('\n')

