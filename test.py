'''
Description: test
Author: Xu Jiaming
Date: 2022-04-28 08:04:38
LastEditTime: 2022-05-02 15:10:38
LastEditors:  
FilePath: test.py
'''
import copy
import random
import numpy as np
import functools
class Node:
    def __init__(self,number):
        self.node = number
        self.parent = number
        self.next = []

class Ti:
    def __init__(self,tree,A):
        self.root = A[0].node
        self.tree = tree
        self.A = A

class Gi:
    def __init__(self,Gi,mmap):
        self.gi = Gi
        self.mmap = mmap

def comp(x, y):
    if(x[0] == y[0]):
        return x[1] > y[1]
    return x[0] > y[0]
 
 
a = [Node(i) for i in range(10)]
b = a[2]
print(a.index(b))  