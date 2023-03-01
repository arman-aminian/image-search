import csv
from typing import List, Dict


def save_as_csv(data: List[Dict], path: str):
    """
    Saves data (list of dicts) on the given path with csv format.
    """
    with open(path, 'w') as file:
        w = csv.DictWriter(file, data[0].keys())
        w.writeheader()
        w.writerows(data)
