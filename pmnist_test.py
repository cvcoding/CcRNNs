import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys

sys.path.append("../../")
from utils import data_generator
from model import TCN
import numpy as np
import argparse
import os
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false', default=True,
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=6,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true', default=False,
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

root = './data/mnist'
batch_size = args.batch_size
n_classes = 10
input_channels = 1
seq_length = int(120 / input_channels)
epochs = args.epochs
steps = 0

print(args)
train_loader, test_loader, valid_loader = data_generator(root, batch_size)

permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
model = TCN(input_channels, n_classes, channel_sizes, args.nhid, kernel_size=kernel_size, dropout=args.dropout)

if args.cuda:
    model.cuda()
    permute = permute.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')


def train(ep):
    global steps
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        # data = data.view(-1, seq_length)    #++++++++++++>>>>>>>>>>>>>>
        if args.permute:
            data = data[:, :, permute]
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss.item() / args.log_interval, steps))
            train_loss = 0


def test():
    model.eval()
    test_loss = 0
    correct = 0
    precision_score_final = 0
    accuracy_score_final = 0
    recall_score_final = 0
    f1_score_final = 0

    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            precision_score_final += precision_score(target.cpu(), pred.squeeze().cpu(), average='weighted')
            accuracy_score_final += accuracy_score(target.cpu(), pred.squeeze().cpu(), normalize=False)  # macro
            recall_score_final += recall_score(target.cpu(), pred.squeeze().cpu(), average='weighted')
            f1_score_final += f1_score(target.cpu(), pred.squeeze().cpu(), average='weighted')

        test_loss /= len(test_loader.dataset)

        total_batch_number = int(len(test_loader.dataset) / batch_size)

        precision_score_final /= total_batch_number
        print('\n[1]test_precision_score_final=', precision_score_final)

        accuracy_score_final /= len(test_loader.dataset)
        print('\n[2]test_accuracy_score_final=', accuracy_score_final)

        recall_score_final /= total_batch_number
        print('\n[3]test_recall_score_final=', recall_score_final)

        f1_score_final /= total_batch_number
        print('\n[4]test_f1_score_final=', f1_score_final)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return test_loss


def valid():
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data.float())
            val_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        val_loss /= len(valid_loader.dataset)
        print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(valid_loader.dataset),
            100. * correct / len(valid_loader.dataset)))
        return val_loss


if __name__ == "__main__":

    best_vloss = 1e8
    all_vloss = []
    for epoch in range(1, epochs + 1):
        train(epoch)
        val_loss = valid()
        test()

        if val_loss < best_vloss:
            with open("model.pt", 'wb') as f:
                print('Save model!\n')
                torch.save(model, f)
            best_vloss = val_loss

        # Anneal the learning rate if the validation loss plateaus
        if epoch > 5 and val_loss >= max(all_vloss[-5:]):
            lr = lr / 2.
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        all_vloss.append(val_loss)

        # if epoch % 10 == 0:
        #     lr /= 10
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

    # Load the best saved model.
    with open("model.pt", 'rb') as f:
        model = torch.load(f)

    test_loss = test()
