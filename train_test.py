import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import numpy as np
from buildFullModel import FullModel
from dataloader import DataLoader
from logger import myLog
from option import opt

Logger = myLog()
program_start = time.time()
Logger.log('Program start time: '+ Logger.getTime(),True)

def logCode(codeFile):
    Logger.log('\n\n'+codeFile+':\n', isShow=False)
    f = open(codeFile, 'r', encoding='utf8')
    Logger.log(f.read(), isShow=False)
    Logger.log('\n\n', isShow=False)
    f.close()
logCode('train_test.py')
logCode('buildFullModel.py')
logCode('dataloader.py')

torch.cuda.set_device(opt.gpu)
dataloader = DataLoader(opt)
RCN = FullModel(opt,dataloader.trainNum)
RCN.cuda()

pdist = nn.PairwiseDistance(2)
metricFunc = nn.HingeEmbeddingLoss(margin=opt.margin, reduction='mean')
idFunc = nn.CrossEntropyLoss(weight=None, reduction='mean')
optimizer = optim.SGD(RCN.parameters(), lr=opt.lr, momentum=opt.momentum)


def train(epoch):
    startTime = time.time()
    RCN.train()
    order = list(range(0, dataloader.trainNum))
    np.random.shuffle(order)
    totalloss = 0.0
    for i in range(0, 2 * dataloader.trainNum):
        inputA, inputB, pair_label, IDA, IDB = dataloader.next()
        optimizer.zero_grad()

        featuresA, soft_featuresA = RCN(inputA)
        featuresB, soft_featuresB = RCN(inputB)
        metric_loss = metricFunc(pdist(featuresA, featuresB), pair_label)
        id_loss = idFunc(soft_featuresA, IDA)+idFunc(soft_featuresB, IDB)
        loss = metric_loss + id_loss
        totalloss = totalloss + float(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(RCN.parameters(), opt.clip)
        optimizer.step()
    Logger.log('epoch {}, loss: {:.4f}, Time cost: {:.4f}'.format(epoch, totalloss, time.time()-startTime),True)


def cmc_compute(test_list, seq_len, visualization=False):
    dist_matrix = np.zeros((dataloader.testNum, dataloader.testNum), dtype=np.float64)
    with torch.no_grad():
        for cr in range(0, 8):
            for fp in range(0, 2):
                for i in range(0, dataloader.testNum):
                    inputA = dataloader.doDataAug(dataloader.dataset[dataloader.testList[i]][dataloader.cams[0]]).cuda()
                    inputB = dataloader.doDataAug(dataloader.dataset[dataloader.testList[i]][dataloader.cams[1]]).cuda()
                    feaA, _ = RCN(inputA)
                    feaB, _ = RCN(inputB)
                    if i == 0:
                        feasA = np.zeros((dataloader.testNum, opt.embeddingSize))
                        feasB = np.zeros((dataloader.testNum, opt.embeddingSize))
                    feasA[i, :] = feaA.cpu().data.numpy()[0]
                    feasB[i, :] = feaB.cpu().data.numpy()[0]

                for i in range(0, dataloader.testNum):
                    for j in range(0, dataloader.testNum):
                        dist_matrix[i, j] = dist_matrix[i, j] + np.linalg.norm(feasA[i, :] - feasB[j, :])
    acc = np.zeros(opt.rankNum)
    tp = 0
    sorted_index = np.argsort(dist_matrix, 0)
    for r in range(0, opt.rankNum):
        tp = 0
        for i in range(0, dataloader.testNum):
            temp = sorted_index[0:r + 1, i]
            if i in temp:
                tp = tp + 1
        acc[r] = tp / float(dataloader.testNum)
    if visualization:
        rankExamples = {}
        for i in range(0, 5):
            temp = sorted_index[0:5 + 1, i]
            rankExamples[dataloader.testList[i]] = [dataloader.testList[p] for p in temp]
        return acc,rankExamples
    else:
        return acc


def test(epoch):
    RCN.eval()
    sl_list = [1, 2, 4, 8, 16, 32, 64, 128]
    for sl in sl_list:
        acc = cmc_compute(dataloader.testList, sl)
        Logger.log('epoch %d test %d images, test accuracy:\n'.format(epoch, sl))
        Logger.log(acc)


for epoch in range(1, opt.epochNum + 1):
    train(epoch)
    seqlen = 16
    if (epoch % 100 == 0):
        startTime = time.time()
        if epoch > 400:
            seqlen = 128
        RCN.eval()
        acc = cmc_compute(dataloader.testList, seqlen)
        Logger.log('epoch {}, {} images, test accuracy:\n'.format(epoch, seqlen),True)
        Logger.log(acc,True)
        Logger.log('Time cost: {:.4f}'.format(time.time()-startTime),True)
    torch.save(RCN, 'RCN.pt')
    if (epoch % 500 ==0):
        Logger.log('Visualization start time: '+ Logger.getTime(),True)
        RCN.eval()
        _,rankExamples = cmc_compute(dataloader.testList, 128,True)
        rankExamples = str(rankExamples)
        f = open('rankExamples'+str(epoch/500)+'.txt','w')
        f.writelines(rankExamples)
        f.close()
        Logger.log('Visualization over time: ' + Logger.getTime(), True)


Logger.log('Program over time: '+ Logger.getTime(),True)