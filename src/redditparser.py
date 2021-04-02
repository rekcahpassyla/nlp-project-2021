# This script will parse Reddit RSS XML into the JSON format
# similar to the Kaggle dataset
import sys
from lxml import etree
import json

if __name__ == '__main__':
    # input file to Reddit XML feed for example as obtained from
    # https://www.reddit.com/r/nottheonion/top.rss?t=year
    inpath = sys.argv[1]
    # output JSON file name
    outpath = sys.argv[2]
    tree = etree.parse(inpath)
    root = tree.getroot()
    jsonout = []
    for item in root.getchildren():
        if item.tag.endswith('entry'):
            for sub in item.getchildren():
                if sub.tag.endswith('title'):
                    jsonitem = {"is_sarcastic": 0}
                    jsonitem["headline"] = sub.text
                    jsonout.append(jsonitem)
    with open(outpath, 'w') as fh:
        json.dump(jsonout, fh, indent=0)
