def parse_dict(data_dict):
    entries = []
    for entry in data_dict:
        for wc_entry in entry["wc"]:
            entries.append({"data": wc_entry,"class": "wc","filename":entry["filename"]})
        for tap_entry in entry["tap water"]:
            entries.append({"data": tap_entry,"class": "tap","filename":entry["filename"]})
    return entries