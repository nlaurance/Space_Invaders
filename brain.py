import lasagne
from numpy import array
import numpy as np
import theano
import theano.tensor as T

from lasagne.layers import InputLayer, DenseLayer, get_all_param_values


# l_in = InputLayer((3, 2))
# l1 = DenseLayer(l_in, num_units=10)
# all_param_values = get_all_param_values(l_in)
#
# print(all_param_values)

# https://martin-thoma.com/lasagne-for-python-newbies/#theano
def build_mlp(input_var):
    l_in = lasagne.layers.InputLayer(shape=(2, 150,),
                                     input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(
        l_in, num_units=200,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    l_hid2 = lasagne.layers.DenseLayer(
        l_hid1, num_units=200,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    l_out = lasagne.layers.DenseLayer(
        l_hid2, num_units=3,
        nonlinearity=lasagne.nonlinearities.softmax)
    return l_out


aa = [(-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
      (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
      (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
      (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
      (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
      (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
      (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
      (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
      (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
      (-1, -1), (177, 82), (227, 82), (277, 82), (327, 82), (377, 82), (427, 82), (477, 82), (527, 82), (577, 82),
      (627, 82), (177, 127), (227, 127), (277, 127), (327, 127), (377, 127), (427, 127), (477, 127), (527, 127),
      (577, 127), (627, 127), (177, 172), (227, 172), (277, 172), (327, 172), (377, 172), (427, 172), (477, 172),
      (527, 172), (577, 172), (627, 172), (177, 217), (227, 217), (277, 217), (327, 217), (377, 217), (427, 217),
      (477, 217), (527, 217), (577, 217), (627, 217), (177, 262), (227, 262), (277, 262), (327, 262), (377, 262),
      (427, 262), (477, 262), (527, 262), (577, 262), (627, 262)]
ff = np.array(aa, dtype=np.dtype('int,int'))
print(ff)
# input_var = T.tensor4('inputs')
# input_var = T.ivector('inputs')
input_var = T.fmatrix('inputs')
network = build_mlp(input_var)

# result = T.ivector('result')

prediction = lasagne.layers.get_output(network, deterministic=True)
result = T.argmax(prediction, axis=1)
predict_fn = theano.function([input_var], result)
o = predict_fn(ff)

print(o)


# all_param_values = get_all_param_values(l_out)
# print(all_param_values)

def predict_label(sample, model='model.npz'):
    input_var = T.tensor4('sample')

    network = build_mlp(input_var)

    with np.load(model) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    lasagne.layers.set_all_param_values(network, param_values)

    prediction = lasagne.layers.get_output(network, deterministic=True)

    result = T.argmax(prediction, axis=1)
    predict_fn = theano.function([input_var], result)
    return predict_fn(sample)
