#!/usr/bin/env python3

import sys
from argparse import ArgumentParser
from collections import defaultdict

def parse_args():
  p = ArgumentParser('Converts integer to word using vocabulary file.')
  p.add_argument(
      '--input',
      type=str, metavar='FILE', required=True, help='source corpus')
  p.add_argument(
      '--output',
      type=str, metavar='FILE', required=True, help='output corpus')
  p.add_argument(
      '--vocab',
      type=str, metavar='FILE', required=True, help='vocabulary file')
  args = p.parse_args()
  return args

def main():
  args = parse_args()
  vocab = defaultdict(lambda: '<unk>')  # unknown ID is converted into '<unk>'
  with open(args.vocab) as vocab_file:
    for line in vocab_file:
      word_id, word, freq = line.split()
      vocab[word_id] = word
  with open(args.input) as input_file, open(args.output, 'w') as output_file:
    for line in input_file:
      words = [vocab[w] for w in line.split()]
      print(' '.join(words), file=output_file)

if __name__ == '__main__':
  main()

