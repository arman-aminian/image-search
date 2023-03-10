import subprocess
import json
from tqdm import tqdm
import requests
from multiprocessing import Pool
from googletrans import Translator
translator = Translator()


def download_nocaps_dataset():
    subprocess.run('curl -o nocaps.json https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json', check=True)
    with open('nocaps.json', 'r') as file:
        dataset = json.loads(file.read())
    return dataset


def clean_dataset(dataset):

    def find_a_comment_for_image(image_id):
        for annotation in dataset['annotations']:
            if annotation['image_id'] == image_id:
                return annotation['caption']

    data = []
    for i in range(1000):
        data.append({
            'url': dataset['images'][i]['coco_url'],
            'comment': find_a_comment_for_image(dataset['images'][i]['id']),
        })

    def download_image(row):
        image_path = '/content/images/' + row['url'].split('/')[-1]
        response = requests.get(row['url'])
        open(image_path, "wb").write(response.content)
    p = Pool(100)
    p.map(download_image, data)

    for i in tqdm(range(len(data))):
        data[i]['translation'] = translator.translate(data[i]['comment'], src='en', dest='fa').text




