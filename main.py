'''
Description: main file
Author: Xu Jiaming
Date: 2022-04-27 17:47:59
LastEditTime: 2022-04-30 00:38:34
LastEditors:  
FilePath: main.py
'''



import copy
import numpy
import random

from torch import le
class Node:
    def __init__(self,number):
        self.node = number
        self.parent = number
        self.next = []

class Ti:
    def __init__(self,tree,A,non_leaf_parent):
        self.root = A[0].node
        self.tree = tree
        self.A = A
        self.nlp = non_leaf_parent

class Gi:
    def __init__(self,Gi,mmap):
        self.gi = Gi
        self.mmap = mmap

       
# 算法1.1产生种群个体    
def generate(F_node,B):

    # Gi种群个体，列表
    Gdi = []
    mmap ={}
    for i in F_node:
        fi_node = Node(i)
        A = [fi_node,]
        # Tree个体列表元素
        Tree = []
        non_leaf_parent = []
        for j in B:
            rnd = random.randint(0,len(A)-1)
            node = Node(j)
            node.parent = rnd
            A[rnd].next = len(A)
            A.append(node)
            new_Anode = copy.deepcopy(A[rnd])
            Tree.append(new_Anode)
            if(A[rnd].parent != A[rnd].node):
                new_non_node = copy.deepcopy(A[A[rnd].parent])
                non_leaf_parent.append(new_non_node)

        ti = Ti(Tree,A,non_leaf_parent)
        mmap[i] = len(Gi)
        Gdi.append(ti)
    gi = Gi(Gdi,mmap)
    return gi


# 算法1：初始化   
def init():
    # 种群大小g
    # F失效节点列表
    # m代表有多少个失效节点
    # n代表总共多少个节点1-n
    # N总节点列表
    # B = N-F
    g = eval(input("Population size:"))  
    n = eval(input("Total nodes:"))
    m = eval(input("Total failure nodes number:"))
    F = []
    print("Input the failure nodes")
    while(m):
        F.append(eval(input()))
        m -= 1
    
    N = [i for i in range(1,n+1)]
    B = list(set(F)-set(N))
    # G 种群
    G = []
    for j in range(g):
        G.append(generate(F,B))
        

# 算法2.1：step3
def step2_3(F,old_p1,old_p2):
    # 交叉子代 crossoffspring
    crossoffspring = []
    number = random.sample(F,len(F)//2)
    p1 = copy.deepcopy(old_p1)
    p2 = copy.deepcopy(old_p2)
    for i in number:
        p1.gi[p1.mmap[i]],p2.gi[p2.mmap[i]] = p2.gi[p2.mmap[i]],p1.gi[p1.mmap[i]]
    crossoffspring.append(p1)
    crossoffspring.append(p2)
    return crossoffspring


# 算法2.2：step4
def step2_4(F,B,old_p1,old_p2):
    t = random.randint(1,len(F))
    f = random.randint(F,t)
    p1 = copy.deepcopy(old_p1)
    p2 = copy.deepcopy(old_p2)
    for i in f:
        p1_fi = p1.gi[p1.mmp[i]]
        p2_fi = p2.gi[p2.mmp[i]]
        b = random.sample(B,1)[0]
        loc1 = 0
        while(loc1 < len(p1_fi.A)):
            if(p1_fi.A[loc1].node == b):
                break
            loc1 += 1
        loc2 = 0
        while(loc2 < len(p2_fi.A)):
            if(p2_fi.A[loc2].node == b):
                break
            loc2 += 1
        p1_fi.A[p1_fi.A[loc1].parent],p2_fi.A[p2_fi.A[loc2].parent] = p2_fi.A[p2_fi.A[loc2].parent],p1_fi.A[p1_fi.A[loc1].parent]
    
    crossoffspring = [p1,p2]
    return crossoffspring

# 算法2：交叉
def cross(G,F,B,pc = 0.8):
    rnd1 = 1
    rnd2 = 2
    while(rnd1==rnd2):
        rnd1 = random.randint(0,len(G)-1)
        rnd2 = random.randint(0,len(G)-1)

    
    parent1 = G[rnd1] 
    parent2 = G[rnd2]

    p = random.randint(0,1000)/1000
    if(p < pc/2):
        # 执行2.1步骤3
        step2_3(F,parent1,parent2)
    elif(p < pc):
        # 执行2.2步骤4
        step2_4(F,B,parent1,parent2)


# 算法3：变异
def mute(G,F,B,pm = 0.15):
    rnd1 = random.randint(0,len(G)-1)
    new_p = copy.deepcopy(G[rnd1])
    p2 = random.randint(0,1000)/1000
    if(p2 < pm):
        t = random.randint(1,len(F))
        f = random.sample(F,t)
        
        mid = set([])
        whole = set([])
        for i in f:
            p_fi = new_p.gi[new_p.mmap[i]]
            for j in p_fi.tree:
                mid.add(j.node)
            for j in p_fi.A:
                whole.add(j.node)
        leaf = whole-mid
        if(len(leaf) != 0):
            L = random.sample(leaf,1)[0]
            for i in f:
                p_fi = new_p.gi[new_p.mmap[i]]
        
                

    else:
