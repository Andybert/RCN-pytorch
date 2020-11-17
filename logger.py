#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
import os
import logging
import time

class myLog(object):
    """docstring for ClassName"""

    def __init__(self, logfilePrefix='log'):
        super(myLog, self).__init__()
        self.logfilePrefix = logfilePrefix
        self.logger = self.get_logger()

    def log(self, content, isShow=True):
        self.logger.info(content)
        if isShow:
            print(content)

    def get_logger(self):
        if not os.path.exists('./log'): os.makedirs('./log')
        FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
        logfile = self.logfilePrefix+'-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
        logfile = os.path.join('log', logfile)
        logging.basicConfig(level=logging.INFO, format=FORMAT, filename=logfile)
        logger = logging.getLogger(__name__)
        # logger.addHandler(logging.StreamHandler())
        return logger

    def getFileName(self):
        splitName = os.path.split(sys.argv[0])
        prefix = splitName[0] + '/' + splitName[1].split('.')[0]
        for i in range(1, 10001):
            logFileName = prefix + '_' + str(i) + '.log'
            if not os.path.exists(logFileName):
                return logFileName

    def getTime(self):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
