# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import *

a = torch.randn(2, 5, 6)
b = [k for k in a]
print(b[0].shape)
a = a.transpose(0,1)
print(a.shape)#5 2 6

w = nn.Linear(18,1,bias=False)
c = torch.randn(3,2,6)
d = []


for i in a:
    k = [w(torch.cat((torch.cat((i,j),dim=1),i*j),dim=1)) for j in c]
    k = torch.stack(k)
    d.append(k)
    #for j in c:
       # k = torch.cat((torch.cat((i,j),dim=1),i*j),dim=1)
        
        #k= w(k)
       # d.append(k)
    #d.append([w(torch.cat((torch.cat((i,j)),i*j))) for j in c])
    
d = torch.stack(d)
    
print(d.shape)

d = torch.randn(15,2,1)
print(d)
d = d.reshape(5,3,2)
print(d[:,0,:])

r = torch.randn(5,2,3)
j = torch.randn(3,6)
k = torch.matmul(r,j)
print(k.shape)

print(r.shape)
r = r.flatten(0)
print(r.shape)


a = torch.randn(5,2,3)
b = torch.zeros(2,3)
c = a*b
print(c.shape)

a = torch.rand(2,4,3)
print(a)

print(torch.max(a,dim=2)[0])
print(torch.prod(torch.max(a,dim=2)[0], dim=1))




a =torch.rand(2,3,4) #[bsz T K]
b = torch.randn(2,3,4)

#print(torch.matmul(a,b).shape)

k = torch.max(a, dim=2)[1]
print(k.shape)
k = k.unsqueeze(2)
print(k)
print(b)
b = b.gather(2,k)
print(b)
print(b.shape)

n = torch.prod(b,dim=1)
print(n.shape)






#print(b.gather(2,k))
#print(torch.index_select(b,2,k))

a = torch.randn(5,6,7)
print(a[1,:4,:].shape)

a = torch.rand(2,3)
b = torch.rand(3,3)
print(a)
print(b)


print(a.unsqueeze(2))