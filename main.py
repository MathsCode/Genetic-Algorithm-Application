'''
Description: main file
Author: Xu Jiaming
Date: 2022-04-27 17:47:59
LastEditTime: 2022-04-28 16:53:17
LastEditors:  
FilePath: main.py
'''




from mmap import mmap
from multiprocessing import parent_process
from cv2 import MOTION_HOMOGRAPHY
import numpy
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
        for j in B:
            rnd = random.randint(0,len(A)-1)
            node = Node(j)
            node.parent = rnd
            A[rnd].next = len(A)
            A.append(node)
            Tree.append(A[rnd])
        ti = Ti(Tree,A)
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
    n = eval(input("Total nodes:"))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    "))  
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
def step2_3(F,p1,p2):
    number = random.sample(F,len(F)//2)
    for i in number:
        


# 算法2：交叉
def cross(G,F,pc = 0.8):
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
    elif(p < pc):
        # 执行2.2步骤4




        
    
        