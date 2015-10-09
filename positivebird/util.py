import csv
import json


def get_config():
    with open('config.json') as config:
        return json.load(config)


def csv_map(file, f):
    results = []

    with open(file, 'r') as fcsv:
        reader = csv.reader(fcsv)
        for row in reader:
            results.append(f(row))

    return results


def csv_each(file, f, encoding='UTF-8'):
    with open(file, 'r', encoding=encoding) as fcsv:
        reader = csv.reader(fcsv)
        for row in reader:
            f(row)