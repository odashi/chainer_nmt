#!/usr/bin/env python3
import math
from argparse import ArgumentParser
from collections import defaultdict


def get_bleu_stats(ref, hyp, N=4):
  stats = defaultdict(int, {'rl': len(ref), 'hl': len(hyp)})
  N = len(hyp) if len(hyp) < N else N
  for n in range(N):
    matched = 0
    possible = defaultdict(int)
    for k in range(len(ref) - n):
      possible[tuple(ref[k : k + n + 1])] += 1
    for k in range(len(hyp) - n):
      ngram = tuple(hyp[k : k + n + 1])
      if possible[ngram] > 0:
        possible[ngram] -= 1
        matched += 1
    stats['d' + str(n + 1)] = len(hyp) - n
    stats['n' + str(n + 1)] = matched
  return stats


def calculate_bleu(stats, N=4):
  np = 0.0
  for n in range(N):
    nn = stats['n' + str(n + 1)]
    if nn == 0:
      return 0.0
    dd = stats['d' + str(n + 1)]
    np += math.log(nn) - math.log(dd)
  bp = 1.0 - stats['rl'] / stats['hl']
  if bp > 0.0: bp = 0.0
  return math.exp(np / N + bp)


def parse_args():
  p = ArgumentParser()
  p.add_argument(
      '--ref',
      type=str, metavar='FILE', required=True,
      help='reference corpus')
  p.add_argument(
      '--hyp',
      type=str, metavar='FILE', required=True,
      help='hypothesis corpus')
  return p.parse_args()


def main():
  args = parse_args()
  stats = defaultdict(int)
  for ref, hyp in zip(open(args.ref), open(args.hyp)):
    for k, v in get_bleu_stats(ref.split(), hyp.split()).items():
      stats[k] += v
  print('%.6f' % calculate_bleu(stats))


if __name__ == '__main__':
  main()
