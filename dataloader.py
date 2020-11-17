import sys
from skimage import io as sio
from skimage.transform import resize
from skimage.color import rgb2yuv
import random
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from ImageFlip import flip as Iflip
from ImageFlip import crop as Icrop


class DataLoader(object):
    """docstring for prepareDataset"""

    def __init__(self, opt):
        super(DataLoader, self).__init__()
        self.opt = opt
        self.sampleInd = 0
        self.iterNum = 0
        self.prepareDataset()
        self.partitionDataset()

    def next(self):
        if self.iterNum % 2 == 0:
            self.getPosSample()
        else:
            self.getNegSample()
        self.iterNum += 1
        inputA = self.doDataAug(self.dataset[self.personA][self.camA][self.startA:self.startA+self.actualSampleSeqLen])
        inputB = self.doDataAug(self.dataset[self.personB][self.camB][self.startB:self.startB+self.actualSampleSeqLen])
        if self.opt.gpu>=0:
            inputA, inputB, self.pair_label, self.IDA, self.IDB = inputA.cuda(),inputB.cuda(),self.pair_label.cuda(),self.IDA.cuda(),self.IDB.cuda()
        return inputA,inputB,self.pair_label,self.IDA,self.IDB

    def loadSeq(self, cam,person):
        seqImgs = self.getSeqImgFiles(os.path.join(self.opt.dir_RGB,cam,person))
        seqLen = len(seqImgs)
        for i in range(0,seqLen):
            filename = seqImgs[i]
            img = sio.imread(os.path.join(self.opt.dir_RGB,cam,person,filename)) * 1.0
            img = rgb2yuv(resize(img, (64, 48), mode='reflect'))

            imgof = sio.imread(os.path.join(self.opt.dir_OF,cam,person,filename)) * 1.0
            imgof = resize(imgof, (64, 48), mode='reflect')

            if i == 0:
                s = img.shape
                imagePixelData = torch.zeros((seqLen, 5, s[0], s[1]),dtype=torch.float)

            for c in range(0, 3):
                d = torch.from_numpy(img[:, :, c])
                m = torch.mean(d, axis=(0, 1))
                v = torch.std(d, axis=(0, 1))
                d = d - m
                d = d / v
                imagePixelData[i, c, :, :] = d
            for c in range(0, 2):
                d = torch.from_numpy(imgof[:, :, c])
                m = torch.mean(d, axis=(0, 1))
                v = torch.std(d, axis=(0, 1))
                d = d - m
                d = d / v
                imagePixelData[i, c + 3, :, :] = d
        return imagePixelData

    def getSeqImgFiles(self, seqRoot):
        seqFiles = []
        for file in os.listdir(seqRoot):
            seqFiles.append(file)
        seqFiles = sorted(seqFiles, key=lambda pd: int(pd[-8:-4]))
        return seqFiles

    def getPersonDirsList(self):
        self.cams = os.listdir(self.opt.dir_RGB)
        personList1 = os.listdir(os.path.join(self.opt.dir_RGB,self.cams[0]))
        personList2 = os.listdir(os.path.join(self.opt.dir_RGB,self.cams[1]))
        commonList = list(set(personList1) & set(personList2))
        if self.opt.debug:
            commonList = commonList[0:10]

        personList = []
        for p in commonList:
            imgs = os.listdir(os.path.join(self.opt.dir_RGB,self.cams[0],p))
            if len(imgs) > 2:
                personList.append(p)

        if len(personList) == 0:
            raise Exception(self.opt.dir_RGB + ' directory does not contain any image files')

        self.personList = sorted(personList, key=lambda pd: int(pd[-3:]))

    def prepareDataset(self):
        self.dataset = {}
        self.getPersonDirsList()
        self.nPersons = len(self.personList)
        for p in self.personList:
            if p not in self.dataset:
                self.dataset[p] = {}
            for cam in self.cams:
                if cam not in self.dataset[p]:
                    self.dataset[p][cam] = {}
                self.dataset[p][cam] = self.loadSeq(cam,p)
                if len(self.dataset[p][cam]) == 0:
                    raise Exception('no dimension')

    def partitionDataset(self):
        splitPoint = int(np.floor(self.nPersons * self.opt.testTrainSplit))
        inds = list(range(0, self.nPersons))
        np.random.shuffle(inds)

        self.trainList = []
        self.testList = []
        for x in range(0, splitPoint):
            self.trainList.append(self.personList[inds[x]])
        for x in range(splitPoint, self.nPersons):
            self.testList.append(self.personList[inds[x]])
        self.trainNum = len(self.trainList)
        self.testNum = len(self.testList)
        print('N train =%d'%(self.trainNum))
        print('N test ==%d'%(self.testNum))

    def getPosSample(self):
        if self.sampleInd >= self.trainNum:
            print(self.sampleInd)
        self.personA,self.personB = self.trainList[self.sampleInd],self.trainList[self.sampleInd]
        self.camA,self.camB = self.cams[0],self.cams[1]
        nSeqA,nSeqB = len(self.dataset[self.personA][self.camA]),len(self.dataset[self.personB][self.camB])

        self.actualSampleSeqLen = int(min(nSeqA, nSeqB, self.opt.seqLen))
        self.startA = random.randint(0, nSeqA - self.actualSampleSeqLen)
        self.startB = random.randint(0, nSeqB - self.actualSampleSeqLen)
        self.pair_label = torch.FloatTensor([1])
        self.IDA = torch.LongTensor([self.sampleInd])
        self.IDB = torch.LongTensor([self.sampleInd])
        self.sampleInd += 1
        if self.sampleInd >= self.trainNum:
            self.sampleInd = 0

    def getNegSample(self):
        permAllPersons = list(range(0, len(self.trainList)))
        np.random.shuffle(permAllPersons)
        self.personA = self.trainList[permAllPersons[1]]
        self.personB = self.trainList[permAllPersons[2]]

        self.camA = self.cams[2-random.randint(1, 2)]
        self.camB = self.cams[2-random.randint(1, 2)]

        nSeqA = len(self.dataset[self.personA][self.camA])
        nSeqB = len(self.dataset[self.personB][self.camB])
        self.actualSampleSeqLen = int(min(nSeqA, nSeqB, self.opt.seqLen))
        self.startA = random.randint(0, nSeqA - self.actualSampleSeqLen)
        self.startB = random.randint(0, nSeqB - self.actualSampleSeqLen)
        self.pair_label = torch.FloatTensor([-1])
        self.IDA = torch.LongTensor([permAllPersons[1]])
        self.IDB = torch.LongTensor([permAllPersons[2]])

    def doDataAug(self, seq):
        cropx = random.randint(0, 7)
        cropy = random.randint(0, 7)
        flip = random.randint(0, 1)
        seqLen = seq.shape[0]
        seqChnls = seq.shape[1]
        seqDim1 = seq.shape[2]
        seqDim2 = seq.shape[3]

        daData = torch.zeros(seqLen, seqChnls, seqDim1 - 8, seqDim2 - 8)
        for t in range(0, seqLen):
            thisFrame = seq[t, :, :, :]
            if flip == 1:
                thisFrame = torch.flip(thisFrame,(2,))

            cropimg = transforms.functional.crop(thisFrame, cropx, cropy, 56, 40)
            daData[t, :, :, :] = cropimg - torch.mean(cropimg, axis=(0, 1, 2))
        return daData
