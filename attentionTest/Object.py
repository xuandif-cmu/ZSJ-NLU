class Object():
    '''
    Top level parent class.
    '''
    def __init__(self, name=""):
        self.__name = name

    def getName(self):
        return self.__name

    def setName(self, name):
        self.__name = name