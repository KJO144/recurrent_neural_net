import numpy as np


def softmax(v):
    e = np.exp(v)
    return e/np.sum(e)


def data_from_text(data_raw):
    chars = list(set(data_raw))
    chars.sort()

    vocab_size = len(chars)
    print('Data has length {} and consist of {} unique characters.'.format(len(data_raw), vocab_size))
    ch_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    data = [ch_to_idx[ch] for ch in data_raw]
    return data, vocab_size, idx_to_char


def rnn_forward(inputs, parameters, h_prev):
    u = parameters['U']
    w = parameters['W']
    bh = parameters['bh']

    seq_length, vocab_size = inputs.shape
    hidden_size = w.shape[1]
    assert(u.shape[1] is vocab_size)
    assert(u.shape[0] is hidden_size)
    assert(w.shape[0] is hidden_size)
    assert(bh.shape == (hidden_size, 1))
    assert(h_prev.shape == (hidden_size, 1))

    h = np.zeros((seq_length, hidden_size, 1))

    for t in range(seq_length):
        xt = inputs[[t]].T
        p = np.dot(u, xt) + np.dot(w, h_prev) + bh
        h[t] = np.tanh(p)
        h_prev = h[t]

    return h


def rnn_backward(yhat, x, target, h, h_prev, parameters):
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
        yhat_minus_y[np.argmax(target[t])] -= 1

        dv += np.outer(yhat_minus_y, h[t])
        dby += yhat_minus_y

        arg = 1 - h[t] * h[t]
        tmat = np.diag(arg[:, 0])
        yv = np.dot(yhat_minus_y.T, v) + passer
        yvt = np.dot(yv, tmat)
        h_tm1 = h[t-1] if t !=0 else h_prev
        dw += np.outer(yvt, h_tm1)
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


def train_rnn(data, vocab_size, seq_length, hidden_size, learning_rate, num_epochs, verbose=False):
    # initialize parameters
    parameters = initialize_parameters(hidden_size, vocab_size)
    data = np.eye(vocab_size)[data]
    data_length = len(data)
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

        for i in range(seqs_per_epoch):
            start = i*seq_length
            end = i*seq_length+seq_length
            end = min(end, data_length-1)

            inputs = data[start:end]
            targets = data[start+1:end+1]

            # forward pass: recurrent layer
            h = rnn_forward(inputs, parameters, h_prev)

            # forward pass: output layer
            v = parameters['V']
            by = parameters['by']
            yhat = {t: softmax(np.dot(v, h[t]) + by) for t in range(seq_length)}

            # compute loss
            losses = [-np.dot(np.log(yhat[t]).T, targets[t]) for t in range(seq_length)]
            loss = np.array(losses).sum()

            # backward pass
            gradients = rnn_backward(yhat, inputs, targets, h, h_prev, parameters)

            # update parameters
            for param, gradient in gradients.items():
                mgradients[param] += gradient * gradient
                parameters[param] -= learning_rate * gradient / np.sqrt(mgradients[param] + 1e-8)
            h_prev = h[len(inputs)-1]
    return parameters
