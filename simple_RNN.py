import numpy as np


def softmax(v):
    e = np.exp(v)
    return e/np.sum(e)


def rnn_forward(inputs, targets, parameters, vocab_size, h_prev):
    h, yhat, x = {}, {}, {}
    u = parameters['U']
    w = parameters['W']
    v = parameters['V']
    by = parameters['by']
    bh = parameters['bh']
    loss = 0
    h[-1] = np.copy(h_prev)

    for t in range(len(inputs)):
        x_index = inputs[t]
        y_index = targets[t]

        # construct a one-hot vector representing the input x
        xt = np.zeros((vocab_size, 1))
        xt[x_index] = 1
        x[t] = xt

        p = np.dot(u, xt) + np.dot(w, h_prev) + bh
        h[t] = np.tanh(p)
        yhat[t] = softmax(np.dot(v, h[t]) + by)
        h_prev = h[t]
        loss -= np.log(np.squeeze(yhat[t][y_index]))
    # loss = np.squeeze(loss)
    return loss, h, yhat, x  # can probably create x outside this fn


def rnn_backward(yhat, x, target, h, parameters):
    '''
    inputs:
        yhat ~
    '''
    v = parameters['V']
    u = parameters['U']
    w = parameters['W']
    bh = parameters['bh']
    by = parameters['by']
    dv, dby, dw, dbh, du = np.zeros_like(v), np.zeros_like(by), np.zeros_like(w), np.zeros_like(bh), np.zeros_like(u)

    passer = np.zeros_like(h[0].T)
    for t in reversed(range(len(target))):
        yhat_minus_y = yhat[t]
        yhat_minus_y[target[t]] -= 1

        dv += np.outer(yhat_minus_y, h[t])
        dby += yhat_minus_y

        arg = 1 - h[t] * h[t]
        tmat = np.diag(arg[:, 0])
        yv = np.dot(yhat_minus_y.T, v) + passer
        yvt = np.dot(yv, tmat)
        dw += np.outer(yvt, h[t-1])
        dbh += yvt.T
        du += np.outer(yvt, x[t])
        passer = np.dot(yvt, w)
    gradients = {'U': du, 'W': dw, 'V': dv, 'by': dby, 'bh': dbh}
    return gradients


def initialize_parameters(hidden_size, vocab_size):
    np.random.seed(0)
    fac = 0.01
    u = np.random.randn(hidden_size, vocab_size) * fac  # input to hidden
    w = np.random.randn(hidden_size, hidden_size) * fac  # hidden to hidden
    v = np.random.randn(vocab_size, hidden_size) * fac  # hidden to output
    bh = np.zeros((hidden_size, 1))  # bias of hidden activation
    by = np.zeros((vocab_size, 1))  # bias of output neuron

    parameters = {'U': u, 'W': w, 'V': v, 'by': by, 'bh': bh}
    return parameters


def generate_sample(parameters, h_prev, seed_index, sample_size):
    np.random.seed(0)

    u = parameters['U']
    v = parameters['V']
    w = parameters['W']
    bh = parameters['bh']
    by = parameters['by']

    vocab_size = u.shape[1]
    x = np.zeros((vocab_size, 1))
    x[seed_index] = 1
    sample = []
    h = h_prev
    for i in range(sample_size):
        h = np.tanh(np.dot(u, x) + np.dot(w, h) + bh)
        yhat = softmax(np.dot(v, h) + by)

        index = np.random.choice(range(vocab_size), p=yhat.ravel())
        sample.append(index)
        x = np.zeros((vocab_size, 1))
        x[index] = 1
    return sample


def train_rnn(data_raw, seq_length, hidden_size, learning_rate, num_epochs, verbose=False):
    chars = list(set(data_raw))
    chars.sort()

    vocab_size = len(chars)
    print('Data has length {} and consist of {} unique characters.'.format(len(data_raw), vocab_size))
    ch_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    data = [ch_to_idx[ch] for ch in data_raw]
    data_length = len(data)

    # initialize parameters
    parameters = initialize_parameters(hidden_size, vocab_size)

    seqs_per_epoch = int(data_length/seq_length)  # will this work if data/seq divides exactly?

    # memory for adagrad
    mgradients = {}
    for param in ['U', 'W', 'V', 'bh', 'by']:
        mgradients[param] = np.zeros_like(parameters[param])

    for epoch in range(num_epochs):
        # initialize hidden state
        h_prev = np.zeros((hidden_size, 1))

        if verbose and epoch != 0:
            print('epoch: {}, loss: {}'.format(epoch, loss))
            # sample_indices = generate_sample(parameters, h_prev, ch_to_idx[' '], 100)
            # sample = ''.join([idx_to_char[idx] for idx in sample_indices])
            # print(sample)

        for i in range(seqs_per_epoch):
            start = i*seq_length
            end = i*seq_length+seq_length
            end = min(end, data_length-1)

            inputs = data[start:end]
            targets = data[start+1:end+1]

            # forward pass
            loss, h, yhat, x = rnn_forward(inputs, targets, parameters, vocab_size, h_prev)

            # backward pass
            gradients = rnn_backward(yhat, x, targets, h, parameters)

            # update parameters
            for param, gradient in gradients.items():
                mgradients[param] += gradient * gradient
                parameters[param] -= learning_rate * gradient / np.sqrt(mgradients[param] + 1e-8)
            h_prev = h[len(inputs)-1]
    return parameters, idx_to_char
