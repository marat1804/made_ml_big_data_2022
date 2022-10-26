#!/usr/bin/env python3
"""mapper for mean"""

import csv
import sys
from statistics import mean

if __name__ == '__main__':
	csv_reader = csv.reader(sys.stdin, delimiter=',')
	prices = []
	for row in csv_reader:
		try:
			prices.append(float(row[9]))
		except ValueError:
			continue

	print(len(prices), mean(prices), sep='\t')
