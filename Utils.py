import pandas as pd
import numpy as np

def parse_dict(data_dict):
    entries = []
    for entry in data_dict:
        for wc_entry in entry["wc"]:
            entries.append({"data": wc_entry,"class": "wc","filename":entry["filename"]})
        for tap_entry in entry["tap water"]:
            entries.append({"data": tap_entry,"class": "tap","filename":entry["filename"]})
    return entries


def get_x_y(data):
    df = pd.read_csv(data)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S.%f').values.astype(np.int64)
    df['time'] = list(map(lambda x: x / 10**9, df['time']))
    return df['time'], df['value']