# coding=utf-8
import argparse
import torch
from torch.autograd import Variable

from dataset import PoetryDataSet
from model import PoetryNet

parser = argparse.ArgumentParser(description='Pytorch 念诗')
parser.add_argument('--data-path', type=str, default='tang.npz', metavar='S',
                    help='the path of data (default: \'tang.npz\'')
parser.add_argument('--model-path', type=str, default='checkpoint/tang_199.pth', metavar='S',
                    help='the path of models (default: \'checkpoint/tang_199.pth\'')
parser.add_argument('--embedding-dim', type=int, default=128, metavar='N',
                    help='input embedding dim vocabulary for model (default: 128)')
parser.add_argument('--hidden-dim', type=int, default=256, metavar='N',
                    help='input hidden dim for model (default: 256)')
parser.add_argument('--max-gen-len', type=int, default=200, metavar='N',
                    help='input the maximum length of generated poem (default: 200)')
parser.add_argument('--start_words', type=str, default='忽如一夜春风来', metavar='N',
                    help='input the start of generated poem (default: \'忽如一夜春风来\')')
parser.add_argument('--background', type=str, default='北风卷地白草折，胡天八月即飞雪。', metavar='N',
                    help='input the background of generated poem (default: \'北风卷地白草折，胡天八月即飞雪。\')')

args = parser.parse_args()
args.start_words = args.start_words.decode("utf-8") # python2.7编码 sha bi
args.background = args.background.decode('utf-8')


#预加载模型
dataset = PoetryDataSet(args.data_path)
word2ix = dataset.word2ix
ix2word = dataset.ix2word
model = PoetryNet(len(word2ix), args.embedding_dim, args.hidden_dim)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(args.model_path))
else:
    model.load_state_dict(torch.load(args.model_path, lambda a, b: a))

results = list(args.start_words)
start_word_len = len(args.start_words)
input = Variable(torch.Tensor([word2ix['<START>']]).view(1, 1).long())
hidden = None

if args.background:
   for word in args.background:
         output, hidden = model(input, hidden)
         input = Variable(input.data.new([word2ix[word]])).view(1, 1)

for i in range(args.max_gen_len):

    output, hidden = model(input, hidden)

    if i < start_word_len:
        w = results[i]
    else:
        top_index = output.data[0].topk(1)[1][0]
        w = ix2word[top_index]
        results.append(w)
    input = Variable(input.data.new([word2ix[results[-1]]])).view(1,1)
    if w == '<EOP>':
        results.pop()
        break
print ''.join(results)