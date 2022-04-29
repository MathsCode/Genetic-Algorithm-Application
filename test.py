'''
Description: test
Author: Xu Jiaming
Date: 2022-04-28 08:04:38
LastEditTime: 2022-04-30 00:31:03
LastEditors:  
FilePath: test.py
'''
import copy
import random

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

a = [1,2]
b = random.sample(a,1)[0]
print(b)