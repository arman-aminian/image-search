import sys
import os
from shutil import copyfile

#sys.path.append("..")  # Adds higher directory to python modules path.

import hazm
import emoji
import re


class Preprocessor:
    
    def __init__(self):
        self._normalizer = hazm.Normalizer()
        self._stopwords = self._load_stopwords('../train/word2vec/resources/stopwords')
        #self._tagger = hazm.POSTagger(model='./resources/postagger.model')

    def _load_stopwords(self, dir_path):
        stopwords = set()
        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)):
                filepath = os.path.join(dir_path, path)
                with open(filepath, encoding="utf8") as f:
                    for line in f:
                        stopwords.add(line.strip())
        return stopwords

    def _upper_repl(self, match):
      """ Convert mask-special tokens to real special tokens """
      return " [" + match.group(1).upper().replace('-', '_') + "] "


    def _convert_emoji_to_text(self, text, delimiters=('[', ']')):
        """ Convert emojis to something readable by the vocab and model """
        text = emoji.demojize(text, delimiters=delimiters)
        return text


    def _clean_html(self, raw_html):
        """ Remove all html tags """
        cleaner = re.compile('<.*?>')
        cleaned = re.sub(cleaner, '', raw_html)
        return cleaned

    def cleaning(
            self,
            text,
            wikipedia=True,
            normalize_cleaning=True,
            half_space_cleaning=True,
            html_cleaning=True,
            username_cleaning=True,
            hashtag_cleaning=True,
            emoji_convert=True,
            no_phone_numbers=True,
            no_numbers=False,
            no_digits=False,
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="",
            replace_with_digit="0",
            replace_with_currency_symbol=""):
        """ A hierarchy of normalization and preprocessing """
        text = text.strip()
        
        if wikipedia:
            # If your data extracted from WikiPedia
            text = text.replace('_', ' ')
            text = text.replace('«', '').replace('»', '')
            text = text.replace('[[', '[').replace(']]', ']')
            text = text.replace('[ [ ', '[').replace(' ] ]', ']')
            text = text.replace(' [ [', ' [').replace('] ] ', '] ')
            text = text.replace(' [ [ ', ' [').replace(' ] ] ', '] ')
            text = text.replace(' . com', '.com').replace('. com', '.com')
            text = text.replace(' . net', '.net').replace('. net', '.net')
            text = text.replace(' . org', '.org').replace('. org', '.org')
            text = text.replace(' . io', '.io').replace('. io', '.io')
            text = text.replace(' . io', '.io').replace('. io', '.io')
            text = text.replace('ه ی', 'ه')
            text = text.replace('هٔ', 'ه')
            text = text.replace('أ', 'ا')

        if username_cleaning:
            text = re.sub(r"\@[\w.-_]+", " ", text)

        if hashtag_cleaning:
            text = text.replace('#', ' ')
            text = text.replace('_', ' ')

        if emoji_convert:
            text = emoji.emojize(text)
            text = self._convert_emoji_to_text(text)

        # cleaning HTML
        if html_cleaning:
            text = self._clean_html(text)

        # normalizing
        if normalize_cleaning:
            text = self._normalizer.normalize(text)

        # removing weird patterns
        weird_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u'\U00010000-\U0010ffff'
            u"\u200d"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\u3030"
            u"\ufe0f"
            u"\u2069"
            u"\u2066"
            u"\u2013"
            u"\u2068"
            u"\u2067"
            "]+", flags=re.UNICODE)

        text = weird_pattern.sub(r'', text)

        # removing extra spaces, hashtags
        text = re.sub("#", "", text)
        # text = re.sub("\s+", " ", text)

        if half_space_cleaning:
            text = text.replace('\u200c', ' ')
            # text = re.sub("\s+", " ", text)

        text = ' '.join([word for word in text.split() if word not in self._stopwords ])

        return text
        

