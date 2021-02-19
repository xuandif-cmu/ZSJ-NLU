from commonVar import STATUS_FAIL
from commonVar import STATUS_OK

from Object import Object
import numpy as np

class OneHotCoderSNIPS(Object):
    def __init__(self, dataLoader, name="OneHotCoderSNIPS"):
        '''
        @param dataLoader: instance of class DataLoaderSnipsJson, providing label set to initialize
                            related parameters.
        @param name: instance name
        '''
        # parent class init
        Object.__init__(self, name)

        # member init
        self.__dataLoader = dataLoader
        self.__classLabSet = self.__dataLoader.getClassLabelSet()     # class label set
        self.__seqLabSet   = self.__dataLoader.getSeqLabelSet()       # sequence label set
        self.__oneHotCoderClassLab = OneHotCoder(labelSet=self.__classLabSet)    # encoder for class label
        self.__oneHotCoderSeqLab   = OneHotCoder(labelSet=self.__seqLabSet)      # encoder for sequence label

        # print information
        print ("[INFO] One hot coder is initialized: class labels number = %d, sequence label number = %d"%(
            len(self.__classLabSet),len(self.__seqLabSet)
        ))

    def encodeClassLabel(self, labelList):
        '''
        encode class labels
        @param labelList: a list of labels, of batch size
        @return: exit_status, codingResultList
        '''
        # sanity check
        if len(labelList) == 0:
            return STATUS_OK, []
        return self.__encodeClassLab(labelList)

    def encodeSeqLabel(self, labelList):
        '''
        encode sequence labels
        @param labelList: a list of labels, of batch size
        @return: exit_status, codingResultList
        '''
        # sanity check
        if len(labelList) == 0:
            return STATUS_OK, []
        return self.__encodeSeqLab(labelList)

    def decodeClassLabel(self, codeList):
        '''
        decode class labels
        @param codeList: a list of one-hot code, of batch size
        @return: exit_status, labelList
        '''
        if len(codeList) == 0:
            return STATUS_OK, []
        return self.__decodeClassLab(codeList)

    def decodeSeqLabel(self, codeList):
        '''
        decode sequence labels
        @param codeList: a list of one-hot code, of batch size
        @return: exit_status, labelList
        '''
        if len(codeList) == 0:
            return STATUS_OK, []
        return self.__decodeSeqLab(codeList)

    def tokenizeSeqLabel(self, seqlLabList):
        '''
        @param seqlLabList: list[N], a list of sequence labels
        @return: list[N], a list of label token. Token means a numerical (integer) representation of labels.
        '''
        return self.__oneHotCoderSeqLab.tokenizeLabelList(seqlLabList)

    def tokenizeClsLabel(self, label):
        '''
        Tokenize given intent label.
        @param label: input intent label
        @return: token of input class label
        '''
        return self.__oneHotCoderClassLab.tokenizeLabel(label)

    def oneHot2Token(self, oneHotCodeList):
        '''
        Transmit one hot code to token.
        @param oneHotCodeList: a list of one hot code. list[code1, code2, code3, ...]
        @return: a list of one host token. List[token1, token2, token3, ...]
        '''
        tokenList = []
        for code in oneHotCodeList:
            tokenList.append(code.index(1))
        return tokenList

    def seqLabel2Token(self, label):
        '''
        Return token, given a label
        @param label: input label, say 'ClassA'
        @return:  labelToken
            @type: int
        '''
        return self.__oneHotCoderSeqLab.label2token(label)

    def clsLab2Token(self, label):
        '''
        Return token, given a label
        @param label: input label, say 'ClassA'
        @return:  labelToken
            @type: int
        '''
        return self.__oneHotCoderClassLab.label2token(label)

    def token2clsLab(self,token):
        '''
        Return label, given a token
        @param token: input token
        @return: a string
        '''
        return self.__oneHotCoderClassLab.token2label(token)

    def tocken2seqLab(self, token):
        '''
        Return sequence label, given a token
        @param token: input token
        @return: a string
        '''
        return self.__oneHotCoderSeqLab.token2label(token)


    def getSequenceLabelList(self):
        '''
        get sequence label list
        @return:  sequence label list
        '''
        return self.__oneHotCoderSeqLab.getLabelList()

    def getClsLabList(self):
        '''
        Get class label list
        @return: class label list
        '''
        return self.__oneHotCoderClassLab.getLabelList()

    def getSequenceLabelNum(self):
        '''
        get sequence label count
        @return: a number
        '''
        return self.__oneHotCoderSeqLab.getLabelNum()

    def getclsLabNum(self):
        '''
        get class label number
        @return: a int
        '''
        return self.__oneHotCoderClassLab.getLabelNum()

    def __encodeClassLab(self, labelList):
        '''
        private function to encode class labels
        @param codeList:
        @return:
        '''
        return self.__oneHotCoderClassLab.encode(labelList=labelList)

    def __encodeSeqLab(self, labelList):
        '''
        private function to encode sequence labels
        @param codeList: Two dimension list, sequence lists: sentence * word
        @return: (exitStatus, result)
        '''
        result = []
        for sentence in labelList:
            rt, tempResult = self.__oneHotCoderSeqLab.encode(sentence)
            if rt == STATUS_OK:
                result.append(tempResult)
                # for debug
                # encode 1's position, not [0,0,0,0,0,1,0,0]
                #result.append([wordCode.index(1) for wordCode in tempResult])
            else:
                print ("[ERROR] OneHotCoderSNIPS. Fail to encode sequence label for:")
                print (sentence)
                return STATUS_FAIL, None
        return STATUS_OK, result

    def __decodeClassLab(self, codeList):
        '''
        private function to decode class labels
        @param codeList:
        @return:
        '''
        return self.__oneHotCoderClassLab.decode(codeList)

    def __decodeSeqLab(self, codeList):
        '''
        private function to decode sequence labels
        @param codeList: Three dimensional lists, sentence * word * 1-hot code
        @return: (exitStatus, result)
        '''
        result = []
        for sentence in codeList:
                rt, tempResult = self.__oneHotCoderSeqLab.decode(sentence)
                if rt == STATUS_OK:
                    result.append(tempResult)
                else:
                    print ("[ERROR] OneHotCoderSNIPS. Fail to decode sequence 1-hot codding for:")
                    print (sentence)
                    return STATUS_FAIL, None
        return STATUS_OK, result

class OneHotCoder(Object):
    def __init__(self, labelSet, name = "OneHotCoder"):
        # parent class init
        Object.__init__(self, name)

        # sanity check
        if (len(labelSet) == 0):
            print ("[ERROR] One hot coder fails to initialize. Input label set size is 0!")
            return STATUS_FAIL

        # member init
        self.__labelList = list(labelSet)        # label list, length N
        self.__labelList.sort()                  # sort label, to avoid run2run difference
        self.__eyeMatrix = np.eye(len(self.__labelList), dtype = int)   # eye matrix, shape = N * N
        self.__label2Token = None        # fill up token dict. For instance, 'entityname' -> 12

        # call function to init
        self.__fillTokenDict()

    def __fillTokenDict(self):
        '''
        Fill up token dict to speed up class name tokenization
        @return:
            self.__label2Token will be filled up
        @attention:
            Impact on self.__label2Token
        '''
        self.__label2Token = {}
        for i, value in enumerate(self.__labelList):
            self.__label2Token[value] = i

    def encode(self, labelList):
        '''
        @param labelList: a list of input label, for example:
                                    ['abc', 'e4a', 'F4', 'Bike']
        @return: exit status, one hot encoding result
        @attention: suppose labelList dimension is 1. Following input is illegal:
                                labelList = ["A","B", ["A","B"],[["A"],["B"]]]
        '''
        # sanity check
        for lab in labelList:
            if not (lab in self.__labelList):
                print("[ERROR] One hot coder fails to encode. label %s is not found in this coder!"%(lab))
                return STATUS_FAIL, None

        # encoding
        return STATUS_OK, [self.__eyeMatrix[self.__label2Token[lab]].tolist() for lab in labelList]

    def decode(self, oneHotCodeList):
        '''
        @param oneHotCodeList: a list of one hot code, for example:
                                    [[1,0,0], [0,1,0], [0,0,1], [1,0,0]]
        @return: exit status, one hot decoding result
                 decode result type: list
                 shape: N * 1, 1 for 1 string
        '''
        # sanity check
        for code in oneHotCodeList:
            if not (code.count(1) == 1):
                print("[ERROR] One hot coder fails to decode. Following one hot code is illegal: ", code)
                return STATUS_FAIL, None

        # find hot position
        oneHotCodeListNp = np.asarray(oneHotCodeList)
        onePosition = np.where(oneHotCodeListNp == 1)

        # sanity check
        # any hot position exceeding label number?
        for pos in onePosition[1]:
            if pos >= len(self.__labelList):
                print("[ERROR] One hot coder fails to decode. label %s is not found in this coder!"%(lab))
                return STATUS_FAIL, None

        return STATUS_OK, [self.__labelList[pos] for pos in onePosition[1]]

    def tokenizeLabelList(self, labList):
        '''
        @param labList: list[N], a list of sequence labels
        @return:
                exit status
                list[N], a list of label token. Token means a numerical (integer) representation of labels.
        '''
        # sanity check
        for label in labList:
            if not label in self.__labelList:
                print("[ERROR] One hot coder fails to tokenize sequence label. label %s is not found in this coder!"%(label))
                return STATUS_FAIL, None

        # tokenize given labels
        return STATUS_OK, [self.__label2Token[label] for label in labList]

    def tokenizeLabel(self, lab):
        '''
        @param lab: label
        @return:
                exit status
                token
        '''
        # sanity check
        if not lab in self.__labelList:
            print("[ERROR] One hot coder fails to tokenize sequence label. label %s is not found in this coder!"%(lab))
            return STATUS_FAIL, None

        # tokenize given labels
        return STATUS_OK, self.__label2Token[lab]

    def label2token(self, label):
        '''
        Return token, given a label
        @param label: input label, say 'ClassA'
        @return:  labelTocken
            @type: int
        '''
        return self.__label2Token[label]

    def token2label(self, token):
        '''
        Return a label, given a token
        @param token: input token
        @return: label
        '''
        return self.__labelList[token]

    def getLabelList(self):
        '''
        get label list
        @return: label list
        '''
        return self.__labelList    # TODO, returning pointer to private member should be avoided

    def getLabelNum(self):
        return len(self.__labelList)
