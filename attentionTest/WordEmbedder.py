from commonVar import STATUS_FAIL
from commonVar import STATUS_OK
from commonVar import STATUS_WARNING
from gensim.models.keyedvectors import KeyedVectors

from Object import Object
import numpy as np

class WordEmbedder(Object):
    def __init__(self, embeddingFilePath, name="WordEmbedder"):
        '''
        @param embeddingFilePath: pre-trained word embedding file path
        @param name:  object name
        '''
        # parent class init
        Object.__init__(self, name=name)

        # member init
        self.__embeddingFilePath = embeddingFilePath
        self.__word2vec = KeyedVectors.load_word2vec_format(self.__embeddingFilePath, binary = False)

        # print information
        self.__printInfo()


    def wordEmbed(self, batchUtterances):
        '''
        @param batchUtterances: batch of utterances to be embedded
        @return: embedded result
        '''
        result = []
        for sentence in batchUtterances:
            sentenceResult = []
            for word in sentence:
                wordEmbeddingNdarray = self.__queryEmbeddingByWord(word)
                sentenceResult.append(wordEmbeddingNdarray.tolist())
            result.append(sentenceResult)
        return result

    def __queryEmbeddingByWord(self, word):
        if word in self.__word2vec.vocab:
            wordIndex = self.__word2vec.vocab[word].index
            return self.__word2vec.syn0[wordIndex]
        else:
            return np.zeros(self.__getEmbeddingDim())

    def __printInfo(self):
        print ("[INFO] WordEmbedding: Using embedding file %s"%(self.__embeddingFilePath))
        print ("[INFO] WordEmbedding: Vocabulary size = %d"%(self.__getVocabSize()))
        print ("[INFO] WordEmbedding: Embedding dim = %d"%(self.__getEmbeddingDim()))

    def __getVocabSize(self):
        return len(self.__word2vec.vocab)

    def __getEmbeddingDim(self):
        return self.__word2vec.vector_size
