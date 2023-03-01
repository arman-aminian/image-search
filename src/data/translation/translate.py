import json
from tqdm import tqdm
from googletrans import Translator
from typing import List, Dict

translator = Translator()


def translate(data: List[Dict], checkpoint_path: str) -> List[Dict]:
    """
    This function translates a given dataset from english to farsi.
    Dataset's type should be a list of dict; the dict should have a field named 'comment' for english text.
    A field named 'translation' will be added to each row for farsi translation.
    Because of slow translation (Google Translate's API has slow speed for free use), this function has a checkpointing feature.
    """

    # sending translation requests in batches of size 300 for being faster
    batch_size = 300
    for i in tqdm(range(len(data), batch_size)):
        english_texts = [data[j]['comment'] for j in range(i, i + batch_size) if j < len(data)]

        # retry if confronting any errors while contacting Google Translate servers
        while True:
            try:
                translations = translator.translate(english_texts, src='en', dest='fa')
                break
            except Exception as e:
                print(e)
        farsi_texts = [t.text for t in translations]

        # adding translation field to data rows
        j = i
        for farsi_text in farsi_texts:
            data[j].update({
                'translation': farsi_text,
            })
            j += 1

        # checkpoint every 10 batches and in the end
        if i % (batch_size * 10) == 0 or j == len(data):
            with open(checkpoint_path, 'w') as file:
                file.write(json.dumps(data))
            print(i + batch_size)

    return data
