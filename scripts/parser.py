from lxml import etree
import json
import re
import pandas as pd

files = ['DailymashRSSHistorical', 'BBCRSSHistorical']
tag_list = ['item', 'item']
is_sarcastic = [1, 0]

for ix, _file in enumerate(files):
    
    PATH = f'datasets/{_file}.xml'
    tree = etree.parse(PATH)
    root = tree.getroot()
    channel = root.getchildren()[0]

    result = []
    for document in tree.xpath(f'//{tag_list[ix]}'):
        # get contents
        title = \
            document.find('title').text if document.find('title') is not None else ''
        description = \
            document.find('description').text if document.find('description').text is not None else ''
        category = \
            document.find('category').text if document.find('category') is not None else ''
        date = \
            document.find('pubDate').text if document.find('date') is not None else ''
        
        # remove paragraph and new line characters
        description = re.sub('</?p[^>]*>', '', description).rstrip()

        result.append(
            [title, description, category, date, is_sarcastic[ix]]
            )
    output = pd.DataFrame(
        result, 
        columns = ['headline', 'description', 'category', 'date', 'is_sarcastic'])

    output.to_json(f'datasets/{_file}.json', orient = 'records', lines = True)


