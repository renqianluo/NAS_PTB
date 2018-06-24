from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import scipy.stats
import tensorflow as tf
from encoder import encoder
from decoder import decoder
import six
import json
import collections

_NUM_SAMPLES = {
  'train' : 10000,
  'test' : 50,
}


# Basic model parameters.

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--model_dir', type=str, default='model')
parser.add_argument('--restore', action='store_true', default=False)
parser.add_argument('--encoder_num_layers', type=int, default=1)
parser.add_argument('--encoder_hidden_size', type=int, default=96)
parser.add_argument('--encoder_emb_size', type=int, default=32)
parser.add_argument('--mlp_num_layers', type=int, default=0)
parser.add_argument('--mlp_hidden_size', type=int, default=32)
parser.add_argument('--decoder_num_layers', type=int, default=1)
parser.add_argument('--decoder_hidden_size', type=int, default=32)
parser.add_argument('--source_length', type=int, default=60)
parser.add_argument('--encoder_length', type=int, default=60)
parser.add_argument('--decoder_length', type=int, default=60)
parser.add_argument('--encoder_dropout', type=float, default=0.0)
parser.add_argument('--encoder_mlp_dropout', type=float, default=0.0)
parser.add_argument('--decoder_dropout', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--encoder_vocab_size', type=int, default=21)
parser.add_argument('--decoder_vocab_size', type=int, default=21)
parser.add_argument('--trade_off', type=float, default=0.5)
parser.add_argument('--train_epochs', type=int, default=1000)
parser.add_argument('--eval_frequency', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--start_decay_step', type=int, default=100)
parser.add_argument('--decay_steps', type=int, default=1000)
parser.add_argument('--decay_factor', type=float, default=0.9)
parser.add_argument('--attention', action='store_true', default=False)
parser.add_argument('--max_gradient_norm', type=float, default=5.0)
parser.add_argument('--beam_width', type=int, default=0)
parser.add_argument('--time_major', action='store_true', default=False)
parser.add_argument('--predict_from_file', type=str, default=None)
parser.add_argument('--predict_to_file', type=str, default=None)
parser.add_argument('--predict_beam_width', type=int, default=0)
parser.add_argument('--predict_lambda', type=float, default=0.1)

SOS=0
EOS=0

def input_fn(params, mode, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  def get_filenames(mode, data_dir):
    """Returns a list of filenames."""
    if mode == 'train':
      return [os.path.join(data_dir, 'encoder.train.input'), os.path.join(data_dir, 'encoder.train.target'),
              os.path.join(data_dir, 'decoder.train.target')]
    else:
      return [os.path.join(data_dir, 'encoder.test.input'), os.path.join(data_dir, 'encoder.test.target'),
              os.path.join(data_dir, 'decoder.test.target')]

  files = get_filenames(mode, data_dir)
  encoder_input_dataset = tf.data.TextLineDataset(files[0])
  encoder_target_dataset = tf.data.TextLineDataset(files[1])
  decoder_target_dataset = tf.data.TextLineDataset(files[2])
  
  dataset = tf.data.Dataset.zip((encoder_input_dataset, encoder_target_dataset, decoder_target_dataset))

  is_training = mode == 'train'

  if is_training:
    dataset = dataset.shuffle(buffer_size=_NUM_SAMPLES['train'])

  def decode_record(encoder_src, encoder_tgt, decoder_tgt): #src:sequence tgt:performance
    sos_id = tf.constant([SOS])
    eos_id = tf.constant([EOS])
    encoder_src = tf.string_split([encoder_src]).values
    encoder_src = tf.string_to_number(encoder_src, out_type=tf.int32)
    encoder_tgt = tf.string_to_number(encoder_tgt, out_type=tf.float32)
    decoder_tgt = tf.string_split([decoder_tgt]).values
    decoder_tgt = tf.string_to_number(decoder_tgt, out_type=tf.int32)
    decoder_src = tf.concat([sos_id ,decoder_tgt[:-1]], axis=0)
    return (encoder_src, encoder_tgt, decoder_src, decoder_tgt)

  dataset = dataset.map(decode_record)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  batched_examples = iterator.get_next()

  encoder_input, encoder_target, decoder_input, decoder_target = batched_examples

  assert encoder_input.shape.ndims == 2
  assert encoder_target.shape.ndims == 1
  while encoder_target.shape.ndims < 2:
    encoder_target = tf.expand_dims(encoder_target, axis=-1)
  assert decoder_input.shape.ndims == 2
  assert decoder_target.shape.ndims == 2
  
  return encoder_input, encoder_target, decoder_input, decoder_target


def create_vocab_tables(vocab_file):
  """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
  vocab_table = lookup_ops.index_table_from_file(
      vocab_file, default_value=0)
  return vocab_table  


def get_train_ops(encoder_train_input, encoder_train_target, decoder_train_input, decoder_train_target, params, reuse=False):
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.constant(params['lr'])
  if params['optimizer'] == "sgd":
    learning_rate = tf.cond(
      global_step < params['start_decay_step'],
      lambda: learning_rate,
      lambda: tf.train.exponential_decay(
              learning_rate,
              (global_step - params['start_decay_step']),
              params['decay_steps'],
              params['decay_factor'],
              staircase=True),
              name="calc_learning_rate")
    opt = tf.train.GradientDescentOptimizer(learning_rate)
  elif params['optimizer'] == "adam":
    assert float(params['lr']) <= 0.001, "! High Adam learning rate %g" % params['lr']
    opt = tf.train.AdamOptimizer(learning_rate)
  elif params['optimizer'] == 'adadelta':
    opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
  tf.summary.scalar("learning_rate", learning_rate)

  my_encoder = encoder.Model(encoder_train_input, encoder_train_target, params, tf.estimator.ModeKeys.TRAIN, 'Encoder')
  encoder_outputs = my_encoder.encoder_outputs
  #encoder_state = my_encoder.encoder_state
  encoder_state = my_encoder.arch_emb
  encoder_state.set_shape([None, params['decoder_hidden_size']])
  encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
  encoder_state = (encoder_state,) * params['decoder_num_layers']
  my_decoder = decoder.Model(encoder_outputs, encoder_state, decoder_train_input, decoder_train_target, params, tf.estimator.ModeKeys.EVAL, 'Decoder')
  encoder_loss = my_encoder.loss
  decoder_loss = my_decoder.loss
    
  total_loss = params['trade_off'] * encoder_loss + (1 - params['trade_off']) * decoder_loss + params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  tf.summary.scalar('training_loss', total_loss)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    gradients, variables = zip(*opt.compute_gradients(total_loss))
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, params['max_gradient_norm'])
    train_op = opt.apply_gradients(
      zip(clipped_gradients, variables), global_step=global_step)
  
  return total_loss, learning_rate, train_op, global_step


def get_test_ops(encoder_test_input, encoder_test_target, decoder_test_input, decoder_test_target, params, reuse=False):
  my_encoder = encoder.Model(encoder_test_input, encoder_test_target, params, mode, 'Encoder')
  encoder_outputs = my_encoder.encoder_outputs
  #encoder_state = my_encoder.encoder_state
  encoder_state = my_encoder.arch_emb
  encoder_state.set_shape([None, params['decoder_hidden_size']])
  encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
  encoder_state = (encoder_state,) * params['decoder_num_layers']
  my_decoder = decoder.Model(encoder_outputs, encoder_state, decoder_test_input, decoder_test_target, params, tf.estimator.ModeKeys.EVAL, 'Decoder')
  encoder_loss = my_encoder.loss
  decoder_loss = my_decoder.loss
    
  total_loss = params['trade_off'] * encoder_loss + (1 - params['trade_off']) * decoder_loss + params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  return total_loss, predict_value, encoder_test_target


def get_predict_ops(encoder_predict_input, params, reuse=False):
  encoder_predict_target = None
  decoder_predict_input = None
  decoder_predict_target = None
  my_encoder = encoder.Model(encoder_predict_input, encoder_predict_target, params, tf.estimator.ModeKeys.PREDICT, 'Encoder')
  encoder_outputs = my_encoder.encoder_outputs
  #encoder_state = my_encoder.encoder_state
  encoder_state = my_encoder.arch_emb
  encoder_state.set_shape([None, params['decoder_hidden_size']])
  encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
  encoder_state = (encoder_state,) * params['decoder_num_layers']
  my_decoder = decoder.Model(encoder_outputs, encoder_state, decoder_predict_input, decoder_predict_target, params, mode, 'Decoder')
  res = my_encoder.infer()
  predict_value = res['predict_value']
  arch_emb = res['arch_emb']
  new_arch_emb = res['new_arch_emb']
  new_arch_outputs = res['new_arch_outputs']
  res = my_decoder.decode()
  sample_id = res['sample_id']

  encoder_state = new_arch_emb
  encoder_state.set_shape([None, params['decoder_hidden_size']])
  encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
  encoder_state = (encoder_state,) * params['decoder_num_layers']
  tf.get_variable_scope().reuse_variables()
  my_decoder = decoder.Model(new_arch_outputs, encoder_state, decoder_preidct_input, decoder_predict_target, params, mode, 'Decoder')
  res = my_decoder.decode()
  new_sample_id = res['sample_id']

  return predict_value, sample_id, new_sample_id


def train(params):
  g = tf.Graph()
  with g.as_default():
    encoder_train_input, encoder_train_target, decoder_train_input, decoder_train_target = input_fn(params, 'train', params['data_dir'], params['batch_size'], None)
    encoder_test_input, encoder_test_target, decoder_test_input, decoder_test_target = input_fn(params, 'test', params['data_dir'], _NUM_SAMPLES['test'], None)
    train_loss, learning_rate, train_op, global_step = get_train_ops(encoder_train_input, encoder_train_target, decoder_train_input, decoder_train_target, params)
    _log_variable_sizes(tf.trainable_variables(), 'Trainable Variables')
    test_loss, test_predict_value, test_ground_truth_value = get_test_ops(encoder_test_input, encoder_test_target, decoder_test_input, decoder_test_target, params, True)
    saver = tf.train.Saver(max_to_keep=10)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
    params['model_dir'], save_steps=params['batches_per_epoch'], saver=saver)
    hooks = [checkpoint_saver_hook]
    tf.logging.info('Starting Session')
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(
      config=config, hooks=hooks, checkpoint_dir=params['model_dir']) as sess:
      start_time = time.time()
      while True:
        run_ops = [
          train_loss,
          learning_rate,
          train_op,
          global_step
        ]
        train_loss_v, learning_rate_v, _, global_step_v = sess.run(run_ops)

        epoch = global_step_v // params['batches_per_epoch'] 
        curr_time = time.time()
        if global_step_v % 100 == 0:
          log_string = "epoch={:<6d} ".format(epoch)
          log_string += "step={:<6d} ".format(global_step_v)
          log_string += "loss={:<6f} ".format(train_loss_v)
          log_string += "learning_rate={:<8.4f} ".format(learning_rate_v)
          log_string += "mins={:<10.2f}".format((curr_time - start_time) / 60)
          tf.logging.info(log_string)
        if global_step_v % params['batches_per_epoch'] == 0: 
          test_ops = [
            test_loss, test_predict_value, test_ground_truth_value
          ]
          test_start_time = time.time()
          test_loss_list = []
          test_predict_value_list = []
          test_ground_truth_value_list = []
          for _ in range(_NUM_IMAGES['test'] // _NUM_IMAGES['test']):
            test_loss_v, test_predict_value_v, test_ground_truth_value_v = sess.run(test_ops)
            test_loss_list.append(test_loss_v.flatten())
            test_predict_value_list.append(test_predict_value_v.flatten())
            test_ground_truth_value_list.append(test_ground_truth_value_v.flatten())   
          predictions_list = np.array(test_predict_value_list)
          targets_list = np.array(test_ground_truth_value_list)
          mse = ((predictions_list -  targets_list) ** 2).mean(axis=0)
          pairwise_acc = pairwise_accuracy(targets_list, predictions_list)
          test_time = time.time() - test_start_time
          log_string =  "Evaluation on test data\n"
          log_string += "epoch={:<6d} ".format(epoch)
          log_string += "step={:<6d} ".format(global_step_v)
          log_string += "test loss={:<6f} ".format(np.mean(test_loss_list))
          log_string += "test pairwise accuracy={:<6f} ".format(pairwise_acc)
          log_string += "test mse={:<6f} ".format(mse)
          log_string += "learning_rate={:<8.6f} ".format(learning_rate_v)
          log_string += "secs={:<10.2f}".format((test_time))
          tf.logging.info(log_string)
        if epoch >= params['train_epochs']:
          break


def test(params):
  g = tf.Graph()
  with g.as_default():
    encoder_test_input, encoder_test_target, decoder_test_input, decoder_test_target = input_fn(params, 'test', params['data_dir'], _NUM_SAMPLES['test'], None)
    test_loss, test_predict_value, test_ground_truth_value = get_test_ops(encoder_test_input, encoder_test_target, decoder_test_input, decoder_test_target, params, True)
    _log_variable_sizes(tf.trainable_variables(), 'Trainable Variables')
    tf.logging.info('Starting Session')
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(
      config=config, checkpoint_dir=params['model_dir']) as sess:
      start_time = time.time()
      while True:
        test_ops = [
          test_loss, test_predict_value, test_ground_truth_value
        ]
        test_start_time = time.time()
        test_loss_list = []
        test_predict_value_list = []
        test_ground_truth_value_list = []
        for _ in range(_NUM_IMAGES['test'] // _NUM_IMAGES['test']):
          test_loss_v, test_predict_value_v, test_ground_truth_value_v = sess.run(test_ops)
          test_loss_list.append(test_loss_v.flatten())
          test_predict_value_list.append(test_predict_value_v.flatten())
          test_ground_truth_value_list.append(test_ground_truth_value_v.flatten())   
        predictions_list = np.array(test_predict_value_list)
        targets_list = np.array(test_ground_truth_value_list)
        mse = ((predictions_list -  targets_list) ** 2).mean(axis=0)
        pairwise_acc = pairwise_accuracy(targets_list, predictions_list)
        test_time = time.time() - test_start_time
        log_string =  "Evaluation on test data\n"
        log_string += "epoch={:<6d} ".format(epoch)
        log_string += "step={:<6d} ".format(global_step_v)
        log_string += "test loss={:<6f} ".format(np.mean(test_loss_list))
        log_string += "test pairwise accuracy={:<6f} ".format(pairwise_acc)
        log_string += "test mse={:<6f} ".format(mse)
        log_string += "learning_rate={:<8.6f} ".format(learning_rate_v)
        log_string += "secs={:<10.2f}".format((test_time))
        tf.logging.info(log_string)

def predict(params):
  g = tf.Graph()
  with g.as_default():
    encoder_predict_input, decoder_predict_target = predict_input_fn(FLAGS.predict_from_file)
    predict_value, sample_id, new_sample_id = get_predict_ops(encoder_predict_input, params)
    tf.logging.info('Starting Session')
    config = tf.ConfigProto(allow_soft_placement=True)
    results, new_ids, perfs = [], [], []
    with tf.train.SingularMonitoredSession(
      config=config, checkpoint_dir=params['model_dir']) as sess:
      while True:
        run_ops = [predict_value, sample_id, new_sample_id]
        predict_value_v, sample_id, new_sample_id = sess.run(run_ops)
        results.append(sample_id)
        new_ids.append(new_sample_id)
        perfs.append(perfs)
    
    if FLAGS.predict_to_file:
      output_filename = predict_to_file
    else:
      output_filename = '%s.result' % FLAGS.predict_from_file

    tf.logging.info('Writing results into {0}'.format(output_filename))
    with tf.gfile.Open(output_filename, 'w') as f:
      for res in results:
        f.write('%s\n' % (res))
    with tf.gfile.Open(output_filename+'.new_arch', 'w') as f:
      for res in new_ids:
        f.write('%s\n' % (res))
    with tf.gfile.Open(output_filename+'.perf', 'w') as f:
      for res in perfs:
        f.write('%s\n' % (res))


def _log_variable_sizes(var_list, tag):
  name_to_var = {v.name: v for v in var_list}
  total_size = 0
  for v_name in sorted(list(name_to_var)):
    v = name_to_var[v_name]
    v_size = int(np.prod(np.array(v.shape.as_list())))
    tf.logging.info("Weight    %s\tshape    %s\tsize    %d",
      v.name[:-2].ljust(80),
      str(v.shape).ljust(20), v_size)
    total_size += v_size
  tf.logging.info("%s Total size: %d", tag, total_size)


def get_params():
  params = vars(FLAGS)

  if FLAGS.restore:
    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
      old_params = json.load(f)
    params.update(old_params)

  return params 

def pairwise_accuracy(la, lb):
  N = len(la)
  assert N == len(lb)
  total = 0
  count = 0
  for i in range(N):
    for j in range(i+1, N):
      #if abs(la[i]-la[j]) <= 0.05 and abs(lb[i]-lb[j]) <=0.05:
      #  count += 1
      #  continue
      if la[i] > la[j] and lb[i] > lb[j]:
        count += 1
      if la[i] < la[j] and lb[i] < lb[j]:
        count += 1
      total += 1
  print('N = {}, total = {}, count = {}'.format(N, total, count))
  return float(count) / total

def main(unparsed):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  with open(os.path.join(FLAGS.data_dir, 'encoder.train.input'), 'r') as f:
    lines = f.read().splitlines()
    _NUM_SAMPLES['train'] = len(lines)
  with open(os.path.join(FLAGS.data_dir, 'encoder.test.input'), 'r') as f:
    lines = f.read().splitlines()
    _NUM_SAMPLES['test'] = len(lines)

  print('Found {} in training set, {} in test set'.format(_NUM_SAMPLES['train'], _NUM_SAMPLES['test']))

  if FLAGS.mode == 'train':
    params = get_params()
    with open(os.path.join(params['model_dir'], 'hparams.json'), 'w') as f:
      json.dump(params, f)
    train(params)  
      
  elif FLAGS.mode == 'test':
    if not os.path.exists(os.path.join(FLAGS.model_dir, 'hparams.json')):
      raise ValueError('No hparams.json found in {0}'.format(FLAGS.model_dir))
    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
      params = json.load(f)
    

  elif FLAGS.mode == 'predict':
    if not os.path.exists(os.path.join(FLAGS.model_dir, 'hparams.json')):
      raise ValueError('No hparams.json found in {0}'.format(FLAGS.model_dir))
    params = vars(FLAGS)
    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
      old_params = json.load(f)
      for k,v in old_params.items():
        if not k.startswith('predict'):
          params[k] = v
    predict(params)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
