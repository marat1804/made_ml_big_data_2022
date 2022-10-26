#!/usr/bin/env python3
"""reducer for mean"""

import sys


if __name__ == '__main__':
	current_mean = 0
	current_size = 0

	for line in sys.stdin:
		chunk_size, chunk_mean = map(float, line.strip().split())
		current_mean = (chunk_size * chunk_mean + current_mean * current_size) \
						/ (chunk_size + current_size)
		current_size += chunk_size

	print(current_mean)
