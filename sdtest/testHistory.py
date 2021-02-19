# -*- coding: utf-8 -*-

import pickle


with open('rst/helloW/20191011_174455.pkl', 'rb') as f:
    perform = pickle.load(f)
    perform = perform['perform']
    print(perform[4]['logavgC'])