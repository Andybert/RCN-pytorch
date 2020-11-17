import argparse

parser = argparse.ArgumentParser(description='RCN')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--cuda', action='store_true',
                    help='enables CUDA training')
parser.add_argument('--gpu', type=int, default=0,
                    help='which GPU to use?')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_RGB', type=str, default='/mnt/041DAC9FB576E669/Datasets/iLIDS-VID/sequences/',
                    help='RGB data directory')
parser.add_argument('--dir_OF', type=str, default='/mnt/041DAC9FB576E669/Datasets/iLIDS-VID/i-LIDS-VID-OF-HVP/sequences/',
                    help='optical flow data directory')
parser.add_argument('--testTrainSplit', type=float, default=0.5,
                    help='proportion of training data')
parser.add_argument('--epochNum', type=int, default=1000,
                    help='training epoch')
parser.add_argument('--rankNum', type=int, default=20,
                    help='number of residual blocks')
parser.add_argument('--seqLen', type=int, default=16,
                    help='the length of each video clip used in the training phase')
parser.add_argument('--nFilters', default='16+32+32',
                    help='RCN Filter number')
parser.add_argument('--embeddingSize', type=int, default=128,
                    help='the dimension of final feature vector')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--clip', type=float, default=5.0,
                    help='clip value')
parser.add_argument('--dropCNN', type=float, default=0.6,
                    help='dropout rate for CNN')
parser.add_argument('--dropRNN', type=float, default=0.6,
                    help='dropout rate for RNN')
parser.add_argument('--margin', type=float, default=2.0,
                    help='margin for contrastive loss')


opt = parser.parse_args()
opt.nFilters = list(map(lambda x: int(x), opt.nFilters.split('+')))

for arg in vars(opt):
     if vars(opt)[arg] == 'True':
          vars(opt)[arg] = True
     elif vars(opt)[arg] == 'False':
          vars(opt)[arg] = False
