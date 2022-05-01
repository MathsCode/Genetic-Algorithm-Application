'''
Description: test
Author: Xu Jiaming
Date: 2022-04-28 08:04:38
LastEditTime: 2022-05-01 22:03:11
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
 
 

a = [ {'a':1,'b':2},{'a':10,'b':10},{'a':2,'b':10}]
a.sort(key=lambda item:item['a'])
print(a)