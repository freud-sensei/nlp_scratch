import sys
sys.path.append('..')
from dataset import ptb

corpus, word2id, id2word = ptb.load_data('train')
# valid, test로도 설정 가능

print('말뭉치 크기:', len(corpus))
print('첫 30개 토큰:', corpus[:30])
print()
print('id2word[15]:', id2word[15])
print('word2id["lexus"]:', word2id["lexus"])