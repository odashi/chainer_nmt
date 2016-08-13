#!/usr/bin/env python3

import sys
from argparse import ArgumentParser
from collections import defaultdict

SPECIAL_TOKENS = ['<unk>', '<s>', '</s>']

def parse_args():
  p = ArgumentParser('Constructs vocabulary file.')
  p.add_argument(
      '--input',
      type=str, metavar='FILE', required=True, help='source corpus')
  p.add_argument(
      '--output',
      type=str, metavar='FILE', required=True, help='vocabulary file')
  p.add_argument(
      '--size',
      type=int, metavar='N', required=True, help='vicabulary size')
  args = p.parse_args()
  assert args.size > len(SPECIAL_TOKENS)
  return args

def main():
  args = parse_args()
  freq = defaultdict(int)
  with open(args.input) as fp:
    for line in fp:
      for word in line.split():
        freq[word] += 1
  freq_sorted = sorted(freq.items(), key=lambda x: x[1], reverse=True)
  freq_sorted = [(w, 0) for w in SPECIAL_TOKENS] + freq_sorted
  with open(args.output, 'w') as fp:
    for i, (key, val) in zip(range(args.size), freq_sorted):
      print('%d\t%s\t%d' % (i, key, val), file=fp)

if __name__ == '__main__':
  main()

