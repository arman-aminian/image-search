import os
import subprocess
from typing import List, Dict


def setup_flickr_dataset(base_path: str, kaggle_username: str, kaggle_key: str) -> List[Dict]:
    """
    This function downloads flickr dataset and returns it's text files in a clean format (list of dicts)
    """

    # setup kaggle
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key

    # download & unzip flickr dataset
    subprocess.run("kaggle datasets download -d hsankesara/flickr-image-dataset", check=True)
    subprocess.run(f"unzip -q {base_path}/flickr-image-dataset.zip", check=True)

    # parse it's csv file (because of an error in the file, we couldn't use common libraries to parse the csv file!)
    with open(f'{base_path}/flickr30k_images/results.csv', 'r') as file:
        data = []
        # load headers
        headers = file.readline()[:-1].split('| ')
        # parse each line and add it to cleaned data
        for line in file:
            row = line[:-1].split('| ')
            try:
                data.append({
                    headers[0]: row[0],
                    headers[1]: row[1],
                    headers[2]: row[2],
                })
            # skip invalid rows
            except Exception as e:
                print(line)
                print(e)

    return data
