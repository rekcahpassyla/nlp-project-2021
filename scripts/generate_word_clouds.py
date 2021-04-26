import sys
sys.path.append('../src')

import inputoutput as io
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re


def plot_cloud(wordcloud, pathname):
    # Set figure size
    plt.figure(figsize=(60, 40))
    # Display image
    plt.imshow(wordcloud)
    # No axis details
    plt.axis("off")
    plt.savefig(pathname)


# get datasets
uk_text, uk_labels = io.get_data('../datasets/raw_data/sarcasm_headlines_dataset_uk.json')
us_text, us_labels = io.get_data('../datasets/raw_data/sarcasm_headlines_dataset_us.json')

# split datasets into sarcastic and non-sarcastic
uk_sarcastic = [headline for i, headline in enumerate(uk_text) if uk_labels[i]]
uk_true = [headline for i, headline in enumerate(uk_text) if not uk_labels[i]]
us_sarcastic = [headline for i, headline in enumerate(us_text) if us_labels[i]]
us_true = [headline for i, headline in enumerate(us_text) if not us_labels[i]]

# join lists of headlines into a single string
uk_sarcastic_string = ' '.join(uk_sarcastic).upper()
uk_true_string = ' '.join(uk_true).upper()
us_sarcastic_string = ' '.join(us_sarcastic).upper()
us_true_string = ' '.join(us_true).upper()

# clean text
uk_sarcastic_string = re.sub(r'==.*?==+', '', uk_sarcastic_string)
uk_true_string = re.sub(r'==.*?==+', '', uk_true_string)
us_sarcastic_string = re.sub(r'==.*?==+', '', us_sarcastic_string)
us_true_string = re.sub(r'==.*?==+', '', us_true_string)

for dataset, pathname in [(uk_sarcastic_string, 'uk_sarcastic_wc.png'),
                          (uk_true_string, 'uk_non_sarcastic_wc.png'),
                          (us_sarcastic_string, 'us_sarcastic_wc.png'),
                          (us_true_string, 'us_non_sarcastic_wc.png')]:

    # generate word cloud
    wordcloud = WordCloud(width=3000,
                          height=2000,
                          random_state=1,
                          background_color='black',
                          colormap='Pastel1',
                          collocations=False,
                          stopwords=STOPWORDS).generate(dataset)

    # plot word cloud
    plot_cloud(wordcloud, '../figures/' + pathname)
