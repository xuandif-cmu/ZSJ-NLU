import json
import re
import copy
import random
from Utils import Utils
from Object import Object

from commonVar import STATUS_FAIL
from commonVar import STATUS_OK
from commonVar import STATUS_WARNING

class DataLoaderSnipsJson(Object):
    '''
    Top class of data set. It's open to user.
    It has ability to manage batch data:
        * data can be retrieved batch by batch
        * it works in random mode or regular mode. In random mode, every batch is composed in a random way. In regular mode,
            all batches are generated in a sequential way.
        * batchSize is tunable
        * it's capable of dividing data into training data and test data
        * TODO: Enable training data and test data ratio tuning.

    It owns two DataSetSnipsJson instances, one for training, the other for testing.
    '''
    def __init__(self, jsonFilePath, batchSize, random = True, name = "DataLoaderSnipsJson", seqLabFormat = 1):
        # parent class initialization
        Object.__init__(self, name=name)

        # member variables initialization
        self.__jsonFilePath = jsonFilePath       # json data file path
        self.__batchSize    = batchSize          # batch size
        self.__random       = random             # randomness of generated batch
        self.__trainFreshDataAbsIndexList   = [] # for train, list of fresh data absolute index, data that has not been sent out
        self.__testFreshDataAbsIndexList    = [] # for test, list of fresh data absolute index, data that has not been sent out
        self.__seqLabelSet                  = None  # sequence label set
        self.__clsLab2CountDict = None              # class label to count dict: label  -> count

        # initialization two datasets, one for training , the other for testing
        self.__trainDataSet = DataSetSnipsJson(self.__jsonFilePath, train = True,
                                               seqLabFormat=seqLabFormat)
        self.__testDataSet = DataSetSnipsJson(self.__jsonFilePath, train = False,
                                              seqLabFormat=seqLabFormat)
        trainDataSize = self.__trainDataSet.getAvailableDataNum()
        testDataSize = self.__testDataSet.getAvailableDataNum()
        print("[INFO] Data loader snips json: train data set size = %d"%(trainDataSize))
        print("[INFO] Data loader snips json: test data set size = %d"%(testDataSize))

        # data data consistency check, between train data set and test data set
        # consistent data class label?
        # consistent sequence class label?

        # prepare to generate batches
        self.resetBatchIteration()

    def getSeqLabelsCountDict(self, train):
        if train == True:
            return self.__trainDataSet.getSeqLabelsCountDict()
        else:
            return self.__testDataSet.getSeqLabelsCountDict()

    def getClassLabelsCountDict(self, train):
        if train == True:
            return self.__trainDataSet.getclsLabCountDict()
        else:
            return self.__testDataSet.getclsLabCountDict()

    def resetBatchIteration(self, train = None):
        '''
        reset batch iteration. It should be called when iteration hit the end.
        @param train:
                None:    reset both training and testing dataset
                True:    reset training dataset
                False:   reset testing dataset
        @return:
                No
        '''
        '''
        Function to reset batch iteration. It should be called after train dataset and test dataset hit
        their end.
        It should be called during data laoder initialization
        @param train: flag to indicate train or test data set to be reset
                        None, reset both train and test data iteration
                        True, reset only train data iteration
                        False, reset only test data iteration
        @return: No
        @attention: It impacts members:
                        self.__trainFreshDataAbsIndexList
                        self.__testFreshDataAbsIndexList
        '''
        if train == None:   # reset, for both train and test
            self.__trainFreshDataAbsIndexList = copy.deepcopy(self.__getAvailableDataIndexList(train=True))
            self.__testFreshDataAbsIndexList = copy.deepcopy(self.__getAvailableDataIndexList(train=False))
        elif train == True:
            self.__trainFreshDataAbsIndexList = copy.deepcopy(self.__getAvailableDataIndexList(train=True))
        else:
            self.__testFreshDataAbsIndexList = copy.deepcopy(self.__getAvailableDataIndexList(train=False))

    def nextBatch(self, train):
        '''
        Retrieve next batch of data: queries, intent labels and sequence labels
        @param train:
                True: retrieve next batch of training data
                False: retrieve next batch of testing data
        @return:
                queries, class/intent label, sequence labels
                Type: all three output are list
                Shape: batchSize * H, H denoting some element, according to returned data type

                if data iteration hits the end, 'None, None, None' will be returned
        '''
        # method: compose subset of fresh index list. Index in the subset is going to be inside next batch.

        # which fresh data index list should be cared here? Train or test?
        if train == True:
            freshDataIndexList = self.__trainFreshDataAbsIndexList
        else:
            freshDataIndexList = self.__testFreshDataAbsIndexList

        if len(freshDataIndexList) == 0:
            return None, None, None

        # compose batchDataIndexList: [0,5,2,3,11,567, 7,...]
        if (len(freshDataIndexList) >= self.__batchSize):   # more than 1 batch left
            if self.__random:   # if in random mode, then randomly compose next batch
                # randomly select a subset of index, of batch size, from fresh index list
                batchDataIndexList = random.sample(freshDataIndexList, k=self.__batchSize)
            else:   # if in sequential mode, sequentially compose next batch
                # sequentially select a subset of index, of batch size, from fresh index list
                batchDataIndexList = freshDataIndexList[0: self.__batchSize]
        else:   # left fresh index count is less than batch size
            # just push all left data into one batch
            batchDataIndexList = copy.deepcopy(freshDataIndexList)

        # remove selected index from fresh index list
        for item in batchDataIndexList:
            freshDataIndexList.remove(item)

        # retrieve data, according to batch index list
        queryList, classLabList, seqLabList = self.__getData(train, batchDataIndexList=batchDataIndexList)

        # sanity check
        len1 = len(queryList)
        len2 = len(classLabList)
        len3 = len(seqLabList)
        if not (len1 == len2 and len1 == len3):
            print ("[ERROR] DataLoaderSnipsJson. Inconsistent return data length.")
            print ("[ERROR] DataLoaderSnipsJson. queryListLength = %d, classLabelListLength = %d, seqLabListLenght = %d."%
                   (len1, len2, len3))
            return None, None, None

        return queryList, classLabList, seqLabList

    def getClassLabelSet(self):
        '''
        @return: classLabSet
                type: set
                Shape: N, N is union(trainDataSet, testDataSet) or trainDataSet, depending on case
        '''
        _, trainClsLabSet = self.__trainDataSet.getClassLabelSet()
        _, testClsLabSet  = self.__testDataSet.getClassLabelSet()
        if not trainClsLabSet == testClsLabSet:
            # if two sequence label set is inconsistent, report warning and union them
            print ("[WARNING] train dataset and test dataset have different class label set:")
            print ("[WARNING] train dataset class label set: ")
            print (trainClsLabSet)
            print ("[WARNING] test dataset class label set: ")
            print (testClsLabSet)
            print ("[WARNING] Union of train dataset and test dataset class labels may be used later.")
            return trainClsLabSet.union(testClsLabSet)
        else:
            # if two sequence label sets are consistent, return any one is okay
            return trainClsLabSet

    def getSeqLabelSet(self):     #TODO: it's a dangerous action, to return a private member
        '''
        retrieve sequence label set
        @return: sequenceLabSet
                type: set
                Shape: N, N is union(trainDataSet, testDataSet) or trainDataSet, depending on case
        @attention:
                self.__seqLabelSet may be impacted
        '''
        # check cache, if it's already cached, return
        if not self.__seqLabelSet == None:
            return self.__seqLabelSet

        # if sequence label set is not cached yet, calculate it and cache it
        trainSeqLabSet = self.__trainDataSet.getSeqLabelSet()
        testSeqLabSet  = self.__testDataSet.getSeqLabelSet()
        if not trainSeqLabSet == testSeqLabSet:
            # if two sequence label set is inconsistent, report warning and union them
            print ("[WARNING] train dataset and test dataset have different sequence label set:")
            print ("[WARNING] train dataset sequence label set: ")
            print (trainSeqLabSet)
            print ("[WARNING] test dataset sequence label set: ")
            print (testSeqLabSet)
            print ("[WARNING] Union of train dataset and test dataset sequence labels may be used later.")
            self.__seqLabelSet = trainSeqLabSet.union(testSeqLabSet)
            return self.__seqLabelSet
        else:
            # if two sequence label sets are consistent, return any one is okay
            self.__seqLabelSet = trainSeqLabSet
            return self.__seqLabelSet

    def getSeqLabelSetSize(self):
        seqLabSet = self.getSeqLabelSet()
        return len(seqLabSet)

    def getClsLabSetSize(self):
        clsLabSet = self.getClassLabelSet()
        return len(clsLabSet)

    def getTrainDataCount(self):
        return self.__trainDataSet.getAvailableDataNum()

    def getTestDataCount(self):
        return self.__testDataSet.getAvailableDataNum()

    def __getAvailableDataIndexList(self, train):
        '''
        get available data index list
        @param train: True to get training data, or, get test data
        @return:
        '''
        if train == True:
            return self.__trainDataSet.getAvailableDataIndexList()
        else:
            return self.__testDataSet.getAvailableDataIndexList()

    def __getData(self, train, batchDataIndexList):
        '''
        get data, including queries, intent labels and sequence labels, given a list of absolute index
        @param train:   True to get training data, or, to get test data
        @param batchDataIndexList: specify data index to retrieve, for example, [6,7,4,1,0] to retrieve data at position
                                    6, 7, 4, 1 and 0.
        @return:
                queries, labels, sequence labels
                Type: All return are lists.
                Shape: len(batchDataindexList) * H, H denoting data shape for different return data
        '''
        if train:
            resultQueries = self.__trainDataSet.getQueryDataList(batchDataIndexList=batchDataIndexList)
            resultClassLabList = self.__trainDataSet.getClassLabDataList(batchDataIndexList=batchDataIndexList)
            resultSeqLabList = self.__trainDataSet.getSeqLabDataList(batchDataIndexList=batchDataIndexList)
        else:
            resultQueries = self.__testDataSet.getQueryDataList(batchDataIndexList=batchDataIndexList)
            resultClassLabList = self.__testDataSet.getClassLabDataList(batchDataIndexList=batchDataIndexList)
            resultSeqLabList = self.__testDataSet.getSeqLabDataList(batchDataIndexList=batchDataIndexList)
        return resultQueries, resultClassLabList, resultSeqLabList

class DataSetSnipsJson(Object):
    '''
    Class to capsulate raw snips data set.
    This class may store modified version of raw snips data.
    This class may filter data and store only a subset of raw data.
    '''
    def __init__(self, jsonFilePath, train, trainRatio = 0.7, testRatio = 0.3, seqLabFormat = 1):
        # parent class initialization
        super().__init__()

        # sanity check
        if not(Utils.isEqualFloat((trainRatio+testRatio),1)):
            print ("[ERROR] DataSetSnipsJson initialization: trainRatio + testRatio != 1.0.\
                    trainRatio = %f, testRatio = %f"%(trainRatio, testRatio))
            return STATUS_FAIL

        # read in json file, raw data, text
        self.__rawData = RawDataSetSnipsJson(jsonFilePath)

        # variable initialize
        self.__simpleSeqLabelList    = None      # simple sequence label list. b-entity_name --> entityname
        self.__simpleSeqLabelNum = 0             # unique simple sequence label number.
        self.__simpleSeqLabelSet     = None      # set of simple sequence labels
        self.__simpleLab2CountDict   = None      # a dict to map simple sequence lab to count
        self.__availableDataIndexList = []       # available data index set. It's a subset of raw data,
                                                 # denoted by a collection of index
        self.__trainRatio = trainRatio           # train ratio, for instance 0.7
        self.__testRatio  = testRatio            # test data ratio, for instance 0.3
        self.__seqLabFormat = seqLabFormat       # sequence label format in this dataset
                                                 # 1: simple format, for example: entityname
                                                 # 0: raw format, for example:    i-entity_name
        self.__clsLabSet = None                  # class label set, it's filled up according to self.__availableDataIndexList
        self.__clsLab2CountDict = None           # a dict to map class label to its count
        # show statistics of raw data
        print("[INFO] DataSetSnipsJson initialization: query number: %d"%(self.getQueryNum()))
        print("[INFO] DataSetSnipsJson initialization: class label number: %d"%(self.getClassLabelNum()))
        print("[INFO] DataSetSnipsJson initialization: sequence label number: %d"%(self.getSeqLabelNum()))

        # generate sequence labels with specified format
        if (self.__seqLabFormat == 1):
            self.__simplifySeqLabel()

        # generate available index set
        # available index set is a list of indexes. These indexes refers to data that is exposed to user
        # for example:
        #       available index list = [0,1,2,4,7,8,11,45]
        #       then, we have only 8 queries, together with their class labels and sequence labels
        #       accessible to user
        self.__generateAvailableIndexList(train, self.__trainRatio, self.__testRatio)

    def getQueryDataList(self, batchDataIndexList):
        '''
        Please refer to document of class RawDataSetSnipsJson.
        @param batchDataIndexList:
        @return:
        '''
        return self.__rawData.getQueryDataList(batchDataIndexList=batchDataIndexList)

    def getClassLabDataList(self, batchDataIndexList):
        '''
        Please refer to document of class RawDataSetSnipsJson.
        @param batchDataIndexList:
        @return:
        '''
        return self.__rawData.getClassLabDataList(batchDataIndexList=batchDataIndexList)

    def __getSimpleSeqLabDataList(self, batchDataIndexList):
        '''
        @param batchDataIndexList:  a list of index, say, [4,3,1,0,11]
        @return:
                a list containing simple sequence labels of index in input list
                type: list
                shape: len(batchDataIndexList) * X, X denoting variable length of different queries
        '''
        return [self.__simpleSeqLabelList[i] for i in batchDataIndexList]

    def getSeqLabelsCountDict(self):
        '''
        get sequence labels count, in dict
        @return: a dict, label --> count
        '''
        if self.__seqLabFormat == 1:    # simple format of sequence format
            return self.__getSimpleSeqLabCountDict()
        else:
            return self.__rawData.getSeqLabCountDict()

    def getclsLabCountDict(self):
        '''
        Get class label count dict: label  --> label count
        @return: a dict:  label --> count
        '''
        # check cache
        if not self.__clsLab2CountDict == None:
            return self.__clsLab2CountDict

        # initialize lab2Count dict
        self.__clsLab2CountDict = {}
        rt, clsLabSet = self.getClassLabelSet()
        if not rt == STATUS_OK:   # return check
            print ("[ERROR] DataSetSnipsJson::getclsLabCountDict(). Fail to get class label count dict due to failure when fetching class label set.")
            return STATUS_FAIL
        for lab in clsLabSet:
            self.__clsLab2CountDict[lab] = 0

        # count
        for index in self.__availableDataIndexList:
            label = self.queryClassLab(index)
            self.__clsLab2CountDict[label] = self.__clsLab2CountDict[label] + 1

        return self.__clsLab2CountDict

    def __getSimpleSeqLabCountDict(self):
        '''
        get simple sequence labels count, in dict
        @return: a dict, label --> count
        '''
        # check cache
        if not self.__simpleLab2CountDict == None:
            return self.__simpleLab2CountDict

        # initialize lab2Count dict
        self.__simpleLab2CountDict = {}
        for lab in self.__simpleSeqLabelSet:
            self.__simpleLab2CountDict[lab] = 0

        # count
        for index in self.__availableDataIndexList:
            for label in self.__simpleSeqLabelList[index]:
                self.__simpleLab2CountDict[label] = self.__simpleLab2CountDict[label] + 1

        return self.__simpleLab2CountDict

    def getSeqLabDataList(self, batchDataIndexList):
        '''
        get sequence labels of index in input list
        @param batchDataIndexList:  a list of index, say [1,5,11,0,47]
        @return:
                a list containing sequence labels of index in input list
                type: list
                shape: len(batchDataIndexList) * X, X denoting variable length of different queries
        @attention:
                self.__seqLabFormat has impact here. It decides sequence label format to return.
        '''
        if self.__seqLabFormat == 1:    # simple format of sequence format
            return self.__getSimpleSeqLabDataList(batchDataIndexList=batchDataIndexList)
        else:
            return self.__rawData.getSeqLabDataList(batchDataIndexList=batchDataIndexList)

    def getAvailableDataIndexList(self):
        return self.__availableDataIndexList

    def getSeqLabelNum(self):
        return self.__rawData.getSeqLabelNum()

    def getClassLabelNum(self):
        return self.__rawData.getClassLabelNum()

    def getQueryNum(self):
        return self.__rawData.getQueryNum()

    def splitDataTwoParts(self, ratioList):
        return self.__rawData.splitDataTwoParts(ratioList=ratioList)

    def __generateAvailableIndexList(self, train, trainRatio, testRatio):
        '''
        generate available index list.
        @param train:
            True for train dataset, False for test dataset
        @return:
            exit status
        @attention:
            __availableDataIndexList will be filled up
        '''
        # if it's for training, first 70% data for each intent label is needed
        # or, last 30% is needed

        # split data
        rt = STATUS_OK
        rt, indexSet1, indexSet2 = self.splitDataTwoParts([trainRatio, testRatio])
        if (rt == STATUS_FAIL):
            print ("[ERROR] DataSetSnipsJson, fail to generate available index list.")
            return STATUS_FAIL

        # sanity check
        # any part is empty?
        if len(indexSet1) == 0 or len(indexSet2) == 0:
            print ("[ERROR] DataSetSnipsJson generate available index list: empty index set is found!")
            return  STATUS_FAIL
        # sum(parts) != totalAmount?
        if (not (len(indexSet1) + len(indexSet2)) == self.getQueryNum()):
            print ("[ERROR] DataSetSnipsJson generate available index list: sum(parts) != totalDataAmout!")
            print ("[ERROR] DataSetSnipsJson generate available index list: total amount = %d!"%(self.getQueryNum()))
            print ("[ERROR] DataSetSnipsJson generate available index list: split1 = %d, split2 = %d!"%(len(indexSet1), len(indexSet2)))
            return  STATUS_FAIL
        # overlap between two parts?
        if not (len((set(indexSet1)) & (set(indexSet2))) == 0):
            print ("[ERROR] DataSetSnipsJson generate available index list: Overlapped slit parts!")
            return  STATUS_FAIL

        # which data to use
        if train:
            self.__availableDataIndexList = indexSet1
        else:
            self.__availableDataIndexList = indexSet2

        # print summary of availabel data
        self.__showDataSummaryInfo(self.__availableDataIndexList)

        return STATUS_OK

    def __showDataSummaryInfo(self, dataIndexList):
        '''
        @param dataIndexList: a list of index of data to be summarized. For example:
                                        dataIndexList = [0,5,2,1,55,77,3]
        @return: No
        @attention: No
        '''

        # var init
        infoSummary = {}
        _, classLabelSet = self.getClassLabelSet()
        for lab in classLabelSet:
            infoSummary[lab] = [0, set()]    # classLab --> (utterance count, sequence label set)

        # get information
        for dataAbsIndex in dataIndexList:
            classLab = self.queryClassLab(dataAbsIndex)
            seqLab = self.querySeqLabList(dataAbsIndex)

            infoSummary[classLab][0] = infoSummary[classLab][0] + 1
            for lab in seqLab:
                infoSummary[classLab][1].add(lab)

        # print summary
        print ("[INFO] data set summary: ")
        for lab in infoSummary:
            print ("[INFO]      Class label %25s, data count = %6d, unique sequence label count = %6d"%(
                    lab,
                    infoSummary[lab][0],
                    len(infoSummary[lab][1])
            ))

    def queryClassLab(self, dataAbsIndex):
        '''
        query class labels for given dataAbsIndex
        @param dataAbsIndex: integer, for example, 5
        @return:
            classLabel
            type: string
        '''
        return self.__rawData.queryClassLab(dataAbsIndex)

    def querySeqLabList(self, dataAbsIndex):
        '''
        query sequence labels for given dataAbsIndex
        @param dataAbsIndex: integer, for example, 5
        @return: seqLaebls
                type: list of string
        '''
        if self.__seqLabFormat == 1:   # simplified version of label format
            return self.__simpleSeqLabelList[dataAbsIndex]
        else:                   # raw version of label format
            return self.__rawData.querySeqLabList(dataAbsIndex)

    def getAvailableDataNum(self):
        return len(self.__availableDataIndexList)

    def getClassLabelSet(self):
        '''
        Get class label set.
        @return: a set of class labels
        '''
        if not self.__clsLabSet == None:    # if self.__clsLabSet is already filled up, return it
            return STATUS_OK, self.__clsLabSet

        # if self.__clsLabSet is not filled up yet, fill it up and return
        if len(self.__availableDataIndexList) == 0:
            print ("[ERROR] DataSetSnipsJson getClassLabelSet(). Fail to find any available data index. Available data index list may not have been filled up.")
            return STATUS_FAIL, _

        # fill up class label set
        self.__clsLabSet = set()
        for index in self.__availableDataIndexList:
            self.__clsLabSet.add(self.queryClassLab(index))

        # return class label set
        return STATUS_OK, self.__clsLabSet


    def getSeqLabelSet(self):
        if self.__seqLabFormat == 1:   # simplified version of label format
            if self.__simpleSeqLabelSet == None:   # fill it up and return
                # loop each element of simple sequence label list
                self.__simpleSeqLabelSet = set()
                for sentence in self.__simpleSeqLabelList:
                    for seqLab in sentence:
                        self.__simpleSeqLabelSet.add(seqLab)
            return self.__simpleSeqLabelSet
        else:
            return self.__rawData.getSeqLabelSet()

    def __simplifySeqLabel(self):
        """
        generate simple sequence label, from raw data.
        BIO symbol is removed. '_' is removed.
        Example: i-entity_name   -->   entityname
        :return:
            Exit status
        """
        if not (self.__simpleSeqLabelList == None):
            return STATUS_OK
        else:
            rt = STATUS_OK
            rt, simpleSeqLabel = self.__rawData.generateSimpleSeqLabelList()
            if rt == STATUS_OK:
                self.__simpleSeqLabelList = simpleSeqLabel
                simpleSeqLabelSet = set()
                for sentence in self.__simpleSeqLabelList:
                    for wordLabel in sentence:
                        simpleSeqLabelSet.add(wordLabel)
                self.__simpleSeqLabelNum = len(simpleSeqLabelSet)
                print("[INFO] DataSet initialization: Get simple sequence label, number: %d" % self.__simpleSeqLabelNum)
            else:
                print("[ERROR] DataSet initialization: fail to simplify sequence labels!\n")
            return rt

    def __getSimpleSeqLabelNum(self):
        return self.__simpleSeqLabelNum

class RawDataSetSnipsJson(Object):
    '''
    Class to store json-formatted SNIPS dataset.
    json file is supposed to contain query sentences, sequence labels and class labels, all in text.

    Example of json file content:
    ----------------------example-----------------
        [
	        {
        		"query": "add don and sherri to my meditate to sounds of nature playlist",
    	    	"seqLabel": "O B-entity_name I-entity_name I-entity_name O B-playlist_owner B-playlist I-playlist I-playlist I-playlist I-playlist O",
	    	    "classLabel": "AddToPlaylist"
	        },
	        {
		        ...
	        }
	            ...
        ]
    ----------------------example---end-----------

    This class stores json content in an ordered way. Three list are used to store queries, class labels
    and sequence labels, respectively, sharing the same index.

    All raw text will be case-lowered after read in.
    '''

    def __init__(self, dataSetPath):
        '''
        Json file will be read in.
        All character will be lower-cased.
        :param dataSetPath: json file data path
        '''
        # parent class initialization
        super().__init__()

        # populate parameters
        self.__dataSetPath = dataSetPath
        print("[INFO] RawDataSetSnipsJson initialization: Reading data from: ", self.__dataSetPath)

        # member variable initialization
        self.__queryNum = -1             # utterance count
        self.__classLabelNum = -1        # class label count
        self.__seqLabelNum = -1          # sequence label, or, slot filling, label count
        self.__classLabList = None       # class label list, queryCount*1 vector
        self.__queryList = None          # query list, 2-dimensional list, queryCount * wordCount
        self.__seqBIOLabList = None      # sequence label list, 2-dimensional list, queryCount * seqLabelCount
        self.__classLabSet = None        # class/intent label set, collecting all uniques class labels
        self.__seqLabSet = None          # sequence/slotFilling label set, collecting all unique sequence labels
        self.__clsLab2CountDict = None   # class label to count dict: label  -> count

        # read file
        with open(self.__dataSetPath) as f:
            rawData = json.load(f)
        print("[INFO] RawDataSetSnipsJson initialization: %d data items are read!" % (len(rawData)))

        # read all data in, in raw text
        # static string. Key names in json file
        jsonQuery = "query"
        jsonSeqLabel = "seqLabel"
        jsonClassLabel = "classLabel"
        # read all query
        self.__queryList = [item[jsonQuery].lower().split(" ") for item in rawData]
        # read all class label and lower their case
        self.__classLabList = [item[jsonClassLabel].lower() for item in rawData]
        # read all BIO label sequence
        self.__seqBIOLabList = [item[jsonSeqLabel].lower().split(" ") for item in rawData]

        # sort data, according to intent labels, for the convenience of later processing
        self.__sortDataByIntentLabel()

        # update statistics: number of queries, class labels and sequence labels
        self.__calculateStatistics()

    def getQueryDataList(self, batchDataIndexList):
        '''
        @param batchDataIndexList:  a list of index, say, [4,3,1,0,11]
        @return:
                a list containing queries of index in input list
                type: list
                shape: len(batchDataIndexList) * X, X denoting variable length of different queries
        '''
        return [self.__queryList[i] for i in batchDataIndexList]

    def getClassLabDataList(self, batchDataIndexList):
        '''
        @param batchDataIndexList:  a list of index, say, [4,3,1,0,11]
        @return:
                a list containing class labels of index in input list
                type: list
                shape: len(batchDataIndexList) * 1
        '''
        return [self.__classLabList[i] for i in batchDataIndexList]

    def getSeqLabDataList(self, batchDataIndexList):
        '''
        @param batchDataIndexList:  a list of index, say, [4,3,1,0,11]
        @return:
                a list containing sequence labels of index in input list
                type: list
                shape: len(batchDataIndexList) * X, X denoting variable length of different queries
        '''
        return [self.__seqBIOLabList[i] for i in batchDataIndexList]

    def __sortDataByIntentLabel(self):
        '''
        sort all data according to class/intent label.
        :return:
            Impact on class members:
                self.__classLabList:   sorted result
                self.__queryList:      sorted result
                self.__seqBIOLabList:  sorted result

            exit status
        '''
        # data sanity check
        numClass =  len(self.__classLabList)
        numQuries = len(self.__queryList)
        numSeqLab = len(self.__seqBIOLabList)
        if not (numClass == numQuries and numQuries == numSeqLab):
            print ("[ERROR] raw data set snips, sort data by intent: inconsistent data lengths!")
            return STATUS_FAIL

        # zip class labels, queries and sequence labels together
        zippedData = zip(self.__classLabList, self.__queryList, self.__seqBIOLabList)

        # sorting
        zippedData = sorted(zippedData)

        # extract sort result
        self.__classLabList = [classLab for classLab, _, _ in zippedData]
        self.__queryList = [query for _, query, _ in zippedData]
        self.__seqBIOLabList = [seqLab for _, _, seqLab in zippedData]
        return STATUS_OK

    def __calculateStatistics(self):
        '''
        calculate statistics of this data set
        :return:
            Impact on class members:
                self.__queryNum
                self.__classLabelNum
                self.__seqLabelNum
        '''
        self.__queryNum = len(self.__queryList)
        self.__classLabelNum = len(set(self.__classLabList))
        self.__calculateSeqLabelNum()

    def __calculateSeqLabelNum(self):
        """
        calculate sequence label number: how many different sequence label is here
        :return:
            Impact on class members:
                self.__seqLabelNum
        """
        seqLabSet = set()
        for sequence in self.__seqBIOLabList:
           for label in sequence:
              seqLabSet.add(label)
        self.__seqLabelNum = len(seqLabSet)

    def getQueryNum(self):
        return self.__queryNum

    def getClassLabelNum(self):
        return self.__classLabelNum

    def getSeqLabelNum(self):
        return self.__seqLabelNum

    def generateSimpleSeqLabelList(self):
        '''
        generate simple sequence label, from raw data. BIO symbol is removed. '_' is removed.
        Example: i-entity_name   -->   entityname
        :return:
            STATUS:
                exit status
                type: int
            simpleSeqLabel:
                simplified sequence labels
                type: 2-dimension list, setence*word
        '''
        simpleSeqLabel = []

        # change i-entity_name into entity_name
        # o is kept
        seqNum = len(self.__seqBIOLabList)
        for i in range(seqNum):
            simpleSeqLabel.append([re.sub("(b|i|o)-", "", word) for word in self.__seqBIOLabList[i]])

        # remove all '-' in sequence label
        seqNum = len(simpleSeqLabel)
        for i in range(seqNum):
            simpleSeqLabel[i] = [re.sub("_", "", label) for label in simpleSeqLabel[i]]

        return STATUS_OK, simpleSeqLabel

    def getClassLabelSet(self):
        """
        get class label set
        @return:
            classLabelSet:
                Type: class 'set'
                Shape: N*1, N denoting unique class label number
        @attention:
                Impact on following members:
                   self.__classLabSet
        """
        if self.__classLabSet == None:
            self.__classLabSet = set(self.__classLabList)
            # sort it before output to avoid run2run difference
            #self.__classLabSet = sorted(self.__classLabSet)
        return self.__classLabSet

    def getSeqLabelSet(self):
        """
        get sequence label set
        @return:
            sequence label set:
                Type: class 'set'
                Shape: N, N denoting unique sequence label count
        @attention:
                Impact on following members:
                   self.__seqLabSet
        """
        if self.__seqLabSet == None:    # if it's None, let's fill it up
            # loop each element of simple sequence label list
            self.__seqLabSet = set()
            for sentence in self.__seqBIOLabList:
                for seqLab in sentence:
                    self.__seqLabSet.add(seqLab)

            # sort it before output to avoid run2run difference
            #self.__seqLabSet = sorted(self.__seqLabSet)
        return self.__seqLabSet

    def queryDataAmountForClass(self, label):
        """
        @param label: label to query.
        @return:
            number of data belongs to given label.
        @attention:
            No impact on member variable.
        """
        return self.__classLabList.count(label)

    def splitDataTwoParts(self, ratioList):
        '''
        @param ratioList: ratioList according to which, data is split. For example:
                                [0.6, 0.4]
        @return:
            (exit_status, partList1, partList2):
                exit_status: ...
                partList1: a list of index, belonging to part1
                partList2: a list of index, belonging to part2
        @attention:
            No impact on members.
        '''
        # data sanity check
        if not(len(ratioList) == 2):
            print ("[ERROR] RawDataSetSnipsJson, split data into two parts: ratio items should be 2,\
                    but it's not!")
            return STATUS_FAIL
        if not Utils.isEqualFloat(sum(ratioList),1):
            print ("[ERROR] RawDataSetSnipsJson, split data into two parts: ratio sum is not 1!")
            return STATUS_FAIL

        # initialize result
        resultPart1 = []
        resultPart2 = []

        # split data
        # calculate data amount for each class/intent label
        dataAmount = {}    # class/intent label name --> data amount
        classLabelSet = self.getClassLabelSet()
        for uniqueLab in classLabelSet:
            dataAmount[uniqueLab] = self.queryDataAmountForClass(uniqueLab)
        # collect available index
        for uniqueLab in classLabelSet:
            # start index of current class/intent label
            indexStart = self.__classLabList.index(uniqueLab)

            # split data amount into two parts. For example: 10 = 4 + 6
            rt, dataAmountSplit = Utils.splitIntByRatio(dataAmount[uniqueLab], ratioList)
            if not(rt == STATUS_OK):
                print ("[ERROR] RawDataSetSnipsJson, fail to split data into two parts!")
                return STATUS_FAIL, None, None

            # extract data amount of part1 and part2
            part1Count= dataAmountSplit[0]
            part2Count = dataAmountSplit[1]

            # assemble index set for part1 and part2
            # part1
            for i in range(part1Count):
                resultPart1.append(indexStart + i)
            #part2
            for i in range(part2Count):
                resultPart2.append(indexStart + part1Count + i)

        # sanity check
        if not ((len(resultPart1) + len(resultPart2)) == self.__queryNum):
            print("[ERROR] RawDataSetSnipsJson, fail to split data into two parts! sum(part) != class label numebr.")
            return STATUS_FAIL, None, None

        # sanity check, more
        if not (((len(resultPart1)/self.__classLabelNum) - ratioList[0]) >= 0.15):
            print("[WARNING] RawDataSetSnipsJson. Split data is not consistent to designed ratio!")
            print("[WARNING] RawDataSetSnipsJson. Designed ratio is: ")
            print (ratioList)
            print("[WARNING] RawDataSetSnipsJson. Split data result is: %d : %d"%(len(resultPart1),
                                                                                  len(resultPart2)))
            return STATUS_FAIL, resultPart1, resultPart2

        # return
        return STATUS_OK, resultPart1, resultPart2

    def queryClassLab(self, dataAbsIndex):
        '''
        query class labels for given dataAbsIndex
        @param dataAbsIndex: integer, for example, 5
        @return:
            classLabel
            type: string
        '''
        return self.__classLabList[dataAbsIndex]

    def querySeqLabList(self, dataAbsIndex):
        '''
        query sequence labels for given dataAbsIndex
        @param dataAbsIndex: integer, for example, 5
        @return: seqLaebls
                type: list of string
        '''
        return self.__seqBIOLabListp[dataAbsIndex]

    def getSeqLabCountDict(self):
        '''
        get sequence label count dict, say, 'B-entity_name' --> 68
        @return: a dict: sequence label name  -->  count
        '''
        return None

    def getClsLabCountDict(self):
        '''
        get simple sequence labels count, in dict
        @return: a dict, label --> count
        '''
        # check cache
        if not self.__clsLab2CountDict == None:
            return self.__clsLab2CountDict

        # initialize lab2Count dict
        self.__clsLab2CountDict = {}
        for lab in self.__classLabList:
            self.__clsLab2CountDict[lab] = 0

        # count
        for clsLab in self.__classLabList:
            self.__clsLab2CountDict[clsLab] = self.__clsLab2CountDict[clsLab] + 1

        return self.__clsLab2CountDict
