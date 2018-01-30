# coding=utf-8
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import PoetryDataSet
from model import PoetryNet

parser = argparse.ArgumentParser(description='Pytorch 学习念诗')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--data-path', type=str, default='tang.npz', metavar='S',
                    help='the path of data (default: \'tang.npz\'')
parser.add_argument('--model-path', type=str, default='checkpoint/', metavar='S',
                    help='the path of models (default: \'checkpoint/\'')
parser.add_argument('--embedding-dim', type=int, default=128, metavar='N',
                    help='input embedding dim vocabulary for model (default: 128)')
parser.add_argument('--hidden-dim', type=int, default=256, metavar='N',
                    help='input hidden dim for model (default: 256)')
parser.add_argument('--print-per-batch', type=int, default=20, metavar='N')
args = parser.parse_args()

dataset = PoetryDataSet(args.data_path)
word2ix = dataset.word2ix
data_loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True)


model = PoetryNet(len(word2ix), args.embedding_dim, args.hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()


for epoch in range(args.epochs):
    for i, data in enumerate(data_loader, 1):

        data = data.long().transpose(1, 0).contiguous()
        optimizer.zero_grad()
        input, target = Variable(data[:-1, :]), Variable(data[1:, :])
        output, _ = model(input)
        loss = criterion(output, target.view(-1))
        loss.backward()
        optimizer.step()

        if i % args.print_per_batch == 0:
            print 'epoch {}, iteration {}, loss = {}'.format(epoch + 1, i, loss.data[0])

    torch.save(model.state_dict(), args.model_path + "{}.pth".format(epoch))



