#!/usr/bin/env python3
"""pure_python.py"""

import csv
import sys
from statistics import mean, variance


if __name__ == '__main__':
    csv_reader = csv.DictReader(sys.stdin, delimiter=',')
    prices = [float(row['price']) for row in csv_reader]

    print(len(prices), mean(prices), variance(prices), sep='\t')
