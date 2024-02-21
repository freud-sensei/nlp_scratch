import sys
sys.path.append('..')
from dataset import sequence

(x_train, y_train), (x_test, y_test) = sequence.load_data('addition.txt', seed=1984)
char2id, id2char = sequence.get_vocab()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_train[0])
print(y_train[0])
print(''.join([id2char[i] for i in x_train[0]]))
print(''.join([id2char[i] for i in y_train[0]]))