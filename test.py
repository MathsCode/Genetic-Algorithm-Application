'''
Description: test
Author: Xu Jiaming
Date: 2022-04-28 08:04:38
LastEditTime: 2022-05-01 11:11:56
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

a = Node(10)
b = Node(20)
t1 = Gi(a,a)
t2 = Gi(b,b)
c = [t1,t2]
b = random.sample(c,1)[0].gi.node
print(b)