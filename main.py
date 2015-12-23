import theano
import numpy
from theano import tensor
from blocks.model import Model
from blocks.bricks import Linear, Tanh
from blocks.bricks.lookup import LookupTable
from ctc import CTC
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import IterableDataset
from fuel.streams import DataStream
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule)
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.graph import ComputationGraph

from dummy_dataset import setup_datastream

floatX = theano.config.floatX


n_epochs = 200
num_input_classes = 5
h_dim = 20
rec_dim = 20
num_output_classes = 4


print('Building model ...')

inputt = tensor.lmatrix('input').T
input_mask = tensor.matrix('input_mask').T
y = tensor.lmatrix('output').T
y_mask = tensor.matrix('output_mask').T
# inputt : T x B
# input_mask : T x B
# y : L x B
# y_mask : L x B

# Linear bricks in
input_to_h = LookupTable(num_input_classes, h_dim, name='lookup')
h = input_to_h.apply(inputt)
# h : T x B x h_dim

# RNN bricks
pre_lstm = Linear(input_dim=h_dim, output_dim=4*rec_dim, name='LSTM_linear')
lstm = LSTM(activation=Tanh(),
            dim=rec_dim, name="rnn")
rnn_out, _ = lstm.apply(pre_lstm.apply(h))

# Linear bricks out 
rec_to_o = Linear(name='rec_to_o',
                  input_dim=rec_dim,
                  output_dim=num_output_classes + 1)
y_hat_pre = rec_to_o.apply(rnn_out)
# y_hat_pre : T x B x C+1

# y_hat : T x B x C+1
y_hat = tensor.nnet.softmax(
    y_hat_pre.reshape((-1, num_output_classes + 1))
).reshape((y_hat_pre.shape[0], y_hat_pre.shape[1], -1))
y_hat.name = 'y_hat'

y_hat_mask = input_mask

# Cost
y_len = y_mask.sum(axis=0)
cost = CTC().apply(y, y_hat, y_len, y_hat_mask).mean()
cost.name = 'CTC'

dl, dl_length = CTC().best_path_decoding(y_hat, y_hat_mask)
L = y.shape[0]
B = y.shape[1]
dl = dl[:L, :]
is_error = tensor.neq(dl, y) * tensor.lt(tensor.arange(L)[:,None], y_len[None,:])
is_error = tensor.switch(is_error.sum(axis=0), tensor.ones((B,)), tensor.neq(y_len, dl_length))

error_rate = is_error.mean()
error_rate.name = 'error_rate'

# Initialization
for brick in [input_to_h, pre_lstm, lstm, rec_to_o]:
    brick.weights_init = IsotropicGaussian(0.01)
    brick.biases_init = Constant(0)
    brick.initialize()

print('Bulding DataStream ...')
ds, stream = setup_datastream(batch_size=10,
                              nb_examples=1000, rng_seed=123,
                              min_out_len=10, max_out_len=20)
valid_ds, valid_stream = setup_datastream(batch_size=10,
                                          nb_examples=1000, rng_seed=456,
                                          min_out_len=10, max_out_len=20)

print('Bulding training process...')
algorithm = GradientDescent(cost=cost,
                            parameters=ComputationGraph(cost).parameters,
                            step_rule=CompositeRule([StepClipping(10.0),
                                                     Scale(0.02)]))
monitor_cost = TrainingDataMonitoring([cost, error_rate],
                                      prefix="train",
                                      after_epoch=True)

monitor_valid = DataStreamMonitoring([cost, error_rate],
                                     data_stream=valid_stream,
                                     prefix="valid",
                                     after_epoch=True)

model = Model(cost)
main_loop = MainLoop(data_stream=stream, algorithm=algorithm,
                     extensions=[monitor_cost, monitor_valid,
                                 FinishAfter(after_n_epochs=n_epochs),
                                 Printing()],
                     model=model)

print('Starting training ...')
main_loop.run()


# vim: set sts=4 ts=4 sw=4 tw=0 et :
