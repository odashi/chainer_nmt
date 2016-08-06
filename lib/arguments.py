def add_file_args(p):
  g = p.add_argument_group('file', 'options to specify input/output files')
  g.add_argument(
      '--train-src',
      type=str, metavar='FILE', required=True,
      help='[in] source corpus for training')
  g.add_argument(
      '--train-trg',
      type=str, metavar='FILE', required=True,
      help='[in] target corpus for training')
  g.add_argument(
      '--dev-src',
      type=str, metavar='FILE', required=True,
      help='[in] source corpus for development test')
  g.add_argument(
      '--dev-trg',
      type=str, metavar='FILE', required=True,
      help='[in] target corpus for development test')
  g.add_argument(
      '--test-src',
      type=str, metavar='FILE', required=True,
      help='[in] source corpus for testing')
  g.add_argument(
      '--test-trg',
      type=str, metavar='FILE', required=True,
      help='[in] target corpus for testing')
  g.add_argument(
      '--output',
      type=str, metavar='DIR', required=True,
      help='[out] output directory')


def add_batch_args(p):
  g = p.add_argument_group('batch', 'options to specify the batch data')
  g.add_argument(
      '--train-batch-size',
      type=int, metavar='INT', required=True,
      help='batch size for training')
  g.add_argument(
      '--test-batch-size',
      type=int, metavar='INT', required=True,
      help='batch size for dev/test')
  g.add_argument(
      '--max-sample-length',
      type=int, metavar='INT', required=True,
      help='maximum sequence length of each training sample')
  g.add_argument(
      '--max-length-ratio',
      type=float, metavar='FLOAT', required=True,
      help='maximum sequence length ratio of each training sample')
  g.add_argument(
      '--max-generation-length',
      type=int, metavar='INT', required=True,
      help='maximum sequence length of each generated sample')


def add_training_args(p):
  g = p.add_argument_group('training', 'options to specify training steps')
  g.add_argument(
      '--total-steps',
      type=int, metavar='INT', required=True,
      help='total number of update steps while training')
  g.add_argument(
      '--eval-interval',
      type=int, metavar='INT', required=True,
      help='number of update steps between each evaluation phase')


def add_optimizer_args(p):
  g = p.add_argument_group('optimizer', 'options to specify the optimizer')
  g.add_argument(
      '--learning-rate',
      type=float, metavar='FLOAT', required=True,
      help='learning late for Adam optimizer')
  g.add_argument(
      '--gradient-clipping',
      type=float, metavar='FLOAT', required=True,
      help='gradient clipping threshold')
  g.add_argument(
      '--weight-decay',
      type=float, metavar='FLOAT', required=True,
      help='weight decay strength')


def add_gpu_args(p):
  g = p.add_argument_group('gpu', 'options to specify GPU usage.')
  g.add_argument(
      '--gpu',
      type=int, metavar='INT', required=True,
      help='GPU ID or -1 (CPU)')


def add_simple_encdec_args(p):
  g = p.add_argument_group('model', 'options to specify the translation model')
  g.add_argument(
      '--src-vocab-size',
      type=int, metavar='INT', required=True,
      help='source vocabulary size')
  g.add_argument(
      '--trg-vocab-size',
      type=int, metavar='INT', required=True,
      help='target vocabulary size')
  g.add_argument(
      '--embed-size',
      type=int, metavar='INT', required=True,
      help='embedding layer size')
  g.add_argument(
      '--hidden-size',
      type=int, metavar='INT', required=True,
      help='RNN hidden layer size')


def add_attention_encdec_args(p):
  g = p.add_argument_group('model', 'options to specify the translation model')
  g.add_argument(
      '--src-vocab-size',
      type=int, metavar='INT', required=True,
      help='source vocabulary size')
  g.add_argument(
      '--trg-vocab-size',
      type=int, metavar='INT', required=True,
      help='target vocabulary size')
  g.add_argument(
      '--embed-size',
      type=int, metavar='INT', required=True,
      help='embedding layer size')
  g.add_argument(
      '--hidden-size',
      type=int, metavar='INT', required=True,
      help='RNN hidden layer size')
  g.add_argument(
      '--atten-size',
      type=int, metavar='INT', required=True,
      help='attention layer size')

