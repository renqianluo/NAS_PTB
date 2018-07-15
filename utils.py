"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.rnn.python.ops import *
from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables  # pylint: disable=unused-import
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import core as core_layers

class WeightNormDense(core_layers.Dense):
    def __init__(self, *args, **kwargs):
        self.weight_norm = kwargs.pop("norm")
        super(WeightNormDense, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(
            min_ndim=2, axes={-1: input_shape[-1].value})
        kernel = self.add_variable(
            'kernel',
            shape=[input_shape[-1].value, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.weight_norm:
            self.g = self.add_variable(
                "wn/g",
                shape=(self.units,),
                initializer=init_ops.ones_initializer(),
                dtype=kernel.dtype,
                trainable=True)
            self.kernel = nn_impl.l2_normalize(kernel, dim=0) * self.g
        else:
            self.kernel = kernel
        if self.use_bias:
            self.bias = self.add_variable(
                'bias',
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True


def weight_norm_dense(
        inputs, units,
        activation=None,
        norm=True,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None):
    layer = WeightNormDense(units,
                  activation=activation,
                  norm=norm,
                  use_bias=use_bias,
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer,
                  kernel_regularizer=kernel_regularizer,
                  bias_regularizer=bias_regularizer,
                  activity_regularizer=activity_regularizer,
                  kernel_constraint=kernel_constraint,
                  bias_constraint=bias_constraint,
                  trainable=trainable,
                  name=name,
                  dtype=inputs.dtype.base_dtype,
                  _scope=name,
                  _reuse=reuse)
    return layer.apply(inputs)

class WeightNormLSTMCell(rnn_cell_impl.RNNCell):
  """Weight normalized LSTM Cell. Adapted from `rnn_cell_impl.LSTMCell`.
    The weight-norm implementation is based on:
    https://arxiv.org/abs/1602.07868
    Tim Salimans, Diederik P. Kingma.
    Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks
    The default LSTM implementation based on:
    http://www.bioinf.jku.at/publications/older/2604.pdf
    S. Hochreiter and J. Schmidhuber.
    "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
    The class uses optional peephole connections, optional cell clipping
    and an optional projection layer.
    The optional peephole implementation is based on:
    https://research.google.com/pubs/archive/43905.pdf
    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for
    large scale acoustic modeling." INTERSPEECH, 2014.
  """

  def __init__(self,
               num_units,
               norm=True,
               use_peepholes=False,
               cell_clip=None,
               initializer=None,
               num_proj=None,
               proj_clip=None,
               forget_bias=1,
               activation=None,
               reuse=None):
    """Initialize the parameters of a weight-normalized LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell
      norm: If `True`, apply normalization to the weight matrices. If False,
        the result is identical to that obtained from `rnn_cell_impl.LSTMCell`
      use_peepholes: bool, set `True` to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(WeightNormLSTMCell, self).__init__(_reuse=reuse)

    self._scope = "wn_lstm_cell"
    self._num_units = num_units
    self._norm = norm
    self._initializer = initializer
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._activation = activation or math_ops.tanh
    self._forget_bias = forget_bias

    self._weights_variable_name = "kernel"
    self._bias_variable_name = "bias"

    if num_proj:
      self._state_size = rnn_cell_impl.LSTMStateTuple(num_units, num_proj)
      self._output_size = num_proj
    else:
      self._state_size = rnn_cell_impl.LSTMStateTuple(num_units, num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def _normalize(self, weight, name):
    """Apply weight normalization.
    Args:
      weight: a 2D tensor with known number of columns.
      name: string, variable name for the normalizer.
    Returns:
      A tensor with the same shape as `weight`.
    """

    output_size = weight.get_shape().as_list()[1]
    g = vs.get_variable(name, [output_size], dtype=weight.dtype)
    return nn_impl.l2_normalize(weight, dim=0) * g

  def _linear(self,
              args,
              output_size,
              norm,
              bias,
              bias_initializer=None,
              kernel_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      norm: bool, whether to normalize the weights.
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      kernel_initializer: starting value to initialize the weight.
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
      args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError("linear is expecting 2D arguments: %s" % shapes)
      if shape[1].value is None:
        raise ValueError("linear expects shape[1] to be provided for shape %s, "
                         "but saw %s" % (shape, shape[1]))
      else:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
      weights = vs.get_variable(
          self._weights_variable_name, [total_arg_size, output_size],
          dtype=dtype,
          initializer=kernel_initializer)
      if norm:
        wn = []
        st = 0
        with ops.control_dependencies(None):
          for i in range(len(args)):
            en = st + shapes[i][1].value
            wn.append(
                self._normalize(weights[st:en, :], name="norm_{}".format(i)))
            st = en

          weights = array_ops.concat(wn, axis=0)

      if len(args) == 1:
        res = math_ops.matmul(args[0], weights)
      else:
        res = math_ops.matmul(array_ops.concat(args, 1), weights)
      if not bias:
        return res

      with vs.variable_scope(outer_scope) as inner_scope:
        inner_scope.set_partitioner(None)
        if bias_initializer is None:
          bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)

        biases = vs.get_variable(
            self._bias_variable_name, [output_size],
            dtype=dtype,
            initializer=bias_initializer)

      return nn_ops.bias_add(res, biases)

  def call(self, inputs, state):
    """Run one step of LSTM.
    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: A tuple of state Tensors, both `2-D`, with column sizes
       `c_state` and `m_state`.
    Returns:
      A tuple containing:
      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.
    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    dtype = inputs.dtype
    num_units = self._num_units
    sigmoid = math_ops.sigmoid
    c, h = state

    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    with vs.variable_scope(self._scope, initializer=self._initializer):

      concat = self._linear(
          [inputs, h], 4 * num_units, norm=self._norm, bias=True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

      if self._use_peepholes:
        w_f_diag = vs.get_variable("w_f_diag", shape=[num_units], dtype=dtype)
        w_i_diag = vs.get_variable("w_i_diag", shape=[num_units], dtype=dtype)
        w_o_diag = vs.get_variable("w_o_diag", shape=[num_units], dtype=dtype)

        new_c = (
            c * sigmoid(f + self._forget_bias + w_f_diag * c) +
            sigmoid(i + w_i_diag * c) * self._activation(j))
      else:
        new_c = (
            c * sigmoid(f + self._forget_bias) +
            sigmoid(i) * self._activation(j))

      if self._cell_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        new_c = clip_ops.clip_by_value(new_c, -self._cell_clip, self._cell_clip)
        # pylint: enable=invalid-unary-operand-type
      if self._use_peepholes:
        new_h = sigmoid(o + w_o_diag * new_c) * self._activation(new_c)
      else:
        new_h = sigmoid(o) * self._activation(new_c)

      if self._num_proj is not None:
        with vs.variable_scope("projection"):
          new_h = self._linear(
              new_h, self._num_proj, norm=self._norm, bias=False)

        if self._proj_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          new_h = clip_ops.clip_by_value(new_h, -self._proj_clip,
                                         self._proj_clip)
          # pylint: enable=invalid-unary-operand-type

      new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
      return new_h, new_state
