#!/bin/bash

# NOTE:
# train/dev/test data are extracted from TANAKA Corpus:
# http://www.edrdg.org/wiki/index.php/Tanaka_Corpus

#
# Make vocabulary
#
mkdir -p sample_data/vocab
./make_vocabulary.py \
  --input sample_data/tok/train.en \
  --output sample_data/vocab/train.en \
  --size 5000
./make_vocabulary.py \
  --input sample_data/tok/train.ja \
  --output sample_data/vocab/train.ja \
  --size 7000

#
# Convert words into word IDs
#
mkdir -p sample_data/wid
for corpus in train dev test; do
  for lang in en ja; do
    ./apply_vocabulary.py \
      --input sample_data/tok/${corpus}.${lang} \
      --output sample_data/wid/${corpus}.${lang} \
      --vocab sample_data/vocab/train.${lang}
  done
done

#
# Split training corpus
# This is an optional process, but required when using large corpus.
#
for lang in en ja; do
  split \
    -d -a 3 -n r/5 \
    sample_data/wid/train.${lang} \
    sample_data/wid/train.${lang}.
done

#
# Start training
# When you use train_attention.py, you should specify also --atten-size opation
# --train-src and --train-trg oaptions can be specified using glob wildcard.
# `sample_data/result/lo` contains training status (e.g. dev/test accumulated
# loss). You can monitor them using `tail -f` command.
# Training scripts do not save model parameters, but output translation results
# of dev/test corpus in every --eval-interval steps.
#
./train_encdec.py \
  --train-src 'sample_data/wid/train.en.???' \
  --train-trg 'sample_data/wid/train.ja.???' \
  --dev-src 'sample_data/wid/dev.en' \
  --dev-trg 'sample_data/wid/dev.ja' \
  --test-src 'sample_data/wid/test.en' \
  --test-trg 'sample_data/wid/test.ja' \
  --output 'sample_data/result' \
  --src-vocab-size 5000 \
  --trg-vocab-size 7000 \
  --embed-size 256 \
  --hidden-size 256 \
  --train-batch-size 64 \
  --test-batch-size 16 \
  --max-sample-length 64 \
  --max-length-ratio 2.0 \
  --max-generation-length 64 \
  --total-steps 5000 \
  --eval-interval 100 \
  --learning-rate 0.0002 \
  --gradient-clipping 2.0 \
  --weight-decay 0.0001 \
  --gpu 0

#
# Restore actual words from word IDs and calculate BLEU scores
# When you use train_encdec.py, test BLEU becomes about 9~12.
# When you use train_attention.py, test BLEU becomes about 14~17.
#
mkdir -p sample_data/eval
for corpus in dev test; do
  for wid_file in `ls sample_data/result/${corpus}.hyp.????????`; do
    word_file=sample_data/eval/`basename ${wid_file}`
    ./restore_words.py \
      --input ${wid_file} \
      --output ${word_file} \
      --vocab sample_data/vocab/train.ja
    ./bleu.py \
      --ref sample_data/tok/${corpus}.ja \
      --hyp ${word_file} \
      > ${word_file}.bleu
  done
done
