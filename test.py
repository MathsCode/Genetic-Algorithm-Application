'''
Description: test
Author: Xu Jiaming
Date: 2022-04-28 08:04:38
LastEditTime: 2022-04-29 21:53:55
LastEditors:  
FilePath: test.py
'''
import copy


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
b = Node(11)
g1 = Gi(a,b)
g2 = copy.deepcopy(g1)
g2.gi.parent = 1
print(g1.gi.parent)