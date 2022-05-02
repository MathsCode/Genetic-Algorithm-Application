'''
Description: main file
Author: Xu Jiaming
Date: 2022-04-27 17:47:59
LastEditTime: 2022-05-02 18:07:32
LastEditors:  
FilePath: main.py
'''



import copy
import functools
import numpy as np
import random

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
            A[rnd].next.append(len(A))
            A.append(node)
            new_Anode = copy.deepcopy(A[rnd])
            Tree.append(new_Anode)
            if(A[rnd].parent != A[rnd].node):
                new_non_node = copy.deepcopy(A[A[rnd].parent])
                non_leaf_parent.append(new_non_node)

        ti = Ti(Tree,A,non_leaf_parent)
        mmap[i] = len(Gdi)
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
    B = list(set(N)-set(F))
    # G 种群
    G = []
    for j in range(g):
        G.append(generate(F,B))
    return g,n,m,G,F,B
        

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
    f = random.sample(F,t)
    p1 = copy.deepcopy(old_p1)
    p2 = copy.deepcopy(old_p2)
    for i in f:
        p1_fi = p1.gi[p1.mmap[i]]
        p2_fi = p2.gi[p2.mmap[i]]
        # print(B)
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
        return step2_3(F,parent1,parent2)
    elif(p < pc):
        # 执行2.2步骤4
        return step2_4(F,B,parent1,parent2)
    else:
        return [parent1,parent2]


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
                L_loc = 0
                while(L_loc < len(p_fi.A)):
                    if(p_fi.A[L_loc].node == L):
                        break
                    L_loc+=1    
                        
                if(len(p_fi.nlp)==0):
                    continue
                else:
                    NLP_loc = random.randint(0,len(p_fi.nlp)-1)
                    p_fi.nlp[NLP_loc].next= L_loc
                    p_fi.A[L_loc].parent = NLP_loc
        return new_p
    else:
        return new_p
# 算法4.1 dfs
def dfs(current,data,A):
    # print(A[0].node)
    # print(current.node)
    # print(current.next)
    if(len(current.next) != 0):
        data[current.node] = 1
        for i in current.next:
            # print(i)
            dfs(A[i],data,A)
            data[current.node] += data[A[i].node]
    else:
        data[current.node] = 1
    return 





# 算法4
def search(totaln,old_indi,bandwidth,reuse):
    # indi 为当前个体
    # bandwidth 为带宽矩阵
    # reuse 表示复用矩阵
    # n为总数
    indi = copy.deepcopy(old_indi)
    for ti in indi.gi:
        for j in ti.A:
            for next in j.next:
                reuse[j.node][ti.A[next].node] += 1
                reuse[ti.A[next].node][j.node] += 1
    max_delay = 0
    is_reuse2delay = False
    node_parent = 0
    node_next = 0

    max_ti = 0
    for ti in indi.gi:
        # print(len(ti.A))
        data_size = np.zeros(ti.A[len(ti.A)-1].node + 1)
        delay = np.zeros((ti.A[len(ti.A)-1].node + 1,ti.A[len(ti.A)-1].node + 1))
        dfs(ti.A[0],data_size,ti.A)
        for j in ti.A:
            for next in j.next:
                delay[j.node][ti.A[next].node] = data_size[ti.A[next].node] / (bandwidth[j.node][ti.A[next].node]/reuse[j.node][ti.A[next].node])
                # print(delay[j.node][ti.A[next].node] )
                if(delay[j.node][ti.A[next].node] > max_delay):
                    max_delay = delay[j.node][ti.A[next].node]
                    max_ti = ti
                    node_parent = j
                    node_next = ti.A[next]
                    if(reuse[j.node][ti.A[next].node] > 1):
                        is_reuse2delay = True
                    elif(reuse[j.node][ti.A[next].node] == 1):
                        is_reuse2delay = False
    

    if(is_reuse2delay == True):
        node_loc = random.randint(0,max_ti.A.index(node_next)-1)
        
        pd = True
        k = 0
        while(pd):
            if(max_ti.A[node_loc].node != node_parent.node and max_ti.A[node_loc].node != node_next.node ):
                if((bandwidth[node_parent.node][node_next.node]/ reuse[node_parent.node][node_next.node]) < (bandwidth[max_ti.A[node_loc].node][node_next.node]/(reuse[max_ti.A[node_loc].node][node_next.node]+1))):
                    node_parent.next.remove(max_ti.A.index(node_next))
                    max_ti.A[node_loc].next.append(max_ti.A.index(node_next))
                    node_next.parent = node_loc
                    pd = False
                else:
                    k += 1
                    if(k >= totaln /10):
                        pd = False
            else:
                node_loc = random.randint(0,max_ti.A.index(node_next)-1)
                k += 1
                if(k >= totaln/10):
                    pd = False
    else:
        #while(max_ti)
        node_loc = random.randint(0,max_ti.A.index(node_next)-1)
        
        pd = True
        k = 0
        while(pd):
            if(max_ti.A[node_loc].node != node_parent.node and max_ti.A[node_loc].node != node_next.node):
                if((bandwidth[node_parent.node][node_next.node]/ reuse[node_parent.node][node_next.node]) < (bandwidth[max_ti.A[node_loc].node][node_next.node]/(reuse[max_ti.A[node_loc].node][node_next.node]+1))):
                    # print(node_parent.next)
                    # print(node_next.node)
                    # print()
                    node_parent.next.remove(max_ti.A.index(node_next))
                    max_ti.A[node_loc].next.append(max_ti.A.index(node_next))
                    node_next.parent = node_loc
                    pd = False
                else:
                    k += 1
                    if(k >= totaln /10):
                        pd = False
            else:
                node_loc = random.randint(0,max_ti.A.index(node_next)-1)
                k += 1
                if(k >= totaln/10):
                    pd = False

    return indi


# 算法4.2 计算个体适应度
def cal_delay(old_indi,bandwidth):
    indi = copy.deepcopy(old_indi)
    reuse = np.zeros(bandwidth.shape)
    for ti in indi.gi:
        for j in ti.A:
            for next in j.next:
                reuse[j.node][ti.A[next].node] += 1
                reuse[ti.A[next].node][j.node] += 1
    max_delay = 0
    is_reuse2delay = False
    node_parent = 0
    node_next = 0
    max_ti = 0
    for ti in indi.gi:
        data_size = np.zeros(ti.A[len(ti.A)-1].node + 1)
        delay = np.zeros((ti.A[len(ti.A)-1].node + 1,ti.A[len(ti.A)-1].node + 1))
        dfs(ti.A[0],data_size,ti.A)
        for j in ti.A:
            for next in j.next:
                delay[j.node][ti.A[next].node] = data_size[ti.A[next].node] / (bandwidth[j.node][ti.A[next].node]/reuse[j.node][ti.A[next].node])
                if(delay[j.node][ti.A[next].node] > max_delay):
                    max_delay = delay[j.node][ti.A[next].node]
                    max_ti = ti
                    node_parent = j
                    node_next = ti.A[next]
                    if(reuse[j.node][ti.A[next].node] > 1):
                        is_reuse2delay = True
                    elif(reuse[j.node][ti.A[next].node] == 1):
                        is_reuse2delay = False
    return max_delay
# 算法5选择
def choose(ini_pop,cross,mutation,localsearch,pop_size,bandwidth):
    max_delay = []
    k = 0
    for i in ini_pop:
        t = {}
        t['delay'] = cal_delay(i,bandwidth)
        t['num'] = k
        t['class'] = 1
        max_delay.append(t)
    k = 0
    for i in cross:
        t = {}
        t['delay'] = cal_delay(i,bandwidth)
        t['num'] = k
        t['class'] = 2
        max_delay.append(t)
    k = 0
    for i in mutation:
        t = {}
        t['delay'] = cal_delay(i,bandwidth)
        t['num'] = k
        t['class'] = 3
        max_delay.append(t)
    k = 0
    for i in localsearch:
        t = {}
        t['delay'] = cal_delay(i,bandwidth)
        t['num'] = k
        t['class'] = 4
        max_delay.append(t)
    max_delay.sort(key=lambda item: item['delay'])
    pop_new = []
    for i in range(pop_size):
        if(max_delay[i]['class'] == 1):
            pop_new.append(copy.deepcopy(ini_pop[max_delay[i]['num']]))
        if(max_delay[i]['class'] == 2):
            pop_new.append(copy.deepcopy(cross[max_delay[i]['num']]))
        if(max_delay[i]['class'] == 3):
            pop_new.append(copy.deepcopy(mutation[max_delay[i]['num']]))
        if(max_delay[i]['class'] == 4):
            pop_new.append(copy.deepcopy(localsearch[max_delay[i]['num']]))
    
    return pop_new,max_delay
        
def main():
    pc = eval(input("交叉概率:"))
    pm = eval(input("变异概率:"))
    t = 0
    # P表示当前种群
    # 初始化操作
    g,n,m,G,F,B = init()
    print(g) 
    print(n) 
    print(m) 
    print(B) 
    print(F)  
    np.set_printoptions(suppress=True)
    bandwidth = np.zeros((n+1,n+1))
    for i in range(n+1):
        bandwidth[i][i] = 0
        for j in range(i+1,n+1):
            rnd = random.randint(0,1000)/10
            if(rnd < 2.7):
                bandwidth[i][j] = random.randint(0,30)
                bandwidth[j][i] = bandwidth[i][j]
            elif(rnd < 14):
                bandwidth[i][j] = random.randint(30,200)
                bandwidth[j][i] = bandwidth[i][j]
            elif(rnd < 40.2):
                bandwidth[i][j] = random.randint(200,500)
                bandwidth[j][i] = bandwidth[i][j]
            elif(rnd < 96.3):
                bandwidth[i][j] = random.randint(500,1000)
                bandwidth[j][i] = bandwidth[i][j]
            else:
                bandwidth[i][j] = random.randint(1000,10000)
                bandwidth[j][i] = bandwidth[i][j]
    print(bandwidth)
    reuse = np.zeros((n+1,n+1))
    P = copy.deepcopy(G)
    k = 1
    tot = 1
    last_delay = {'delay':0,'class':0,'num':0}
    while(k <= 50  and tot <= 100):
        print("TOT:",tot,"Num:",k)
        # 交叉操作
        crossoffspring = cross(P,F,B,pc)

        # 变异操作
        mutationoffspring = []
        mutationoffspring.append(mute(P,F,B,pm))
        
        # 局部搜索
        sum = P+crossoffspring+mutationoffspring 
        t = random.randint(0,len(sum)-1)
        pop_search = random.sample(sum,t)
        localsearchoffspring = []
        for i in pop_search:
            localsearchoffspring.append(copy.deepcopy(search(n,i,bandwidth,reuse)))
        pop_new,max_delay = choose(P,crossoffspring,mutationoffspring,localsearchoffspring,g,bandwidth)
        P = copy.deepcopy(pop_new)
        if(last_delay['delay'] == max_delay[0]['delay'] and last_delay['num'] == max_delay[0]['num'] and last_delay['class'] == max_delay[0]['class']):
            k+=1
        else:
            last_delay['delay'] = max_delay[0]['delay'] 
            last_delay['num'] = max_delay[0]['num'] 
            last_delay['class'] = max_delay[0]['class']
            k = 0
        tot+= 1
        print(max_delay[0]['delay'])
        # print(max_delay[0]['num'])
        # print(max_delay[0]['class'])
        if(max_delay[0]['class'] == 1):
            tree = 0
            for i in P[max_delay[0]['num']].gi:
                print("################################")
                tree += 1
                print("修复树:",tree)
                for j in i.A:
                    print("------------------")
                    print("当前节点值：",j.node)
                    print("当前节点的父节点值：",i.A[j.parent].node)
                    lst = []
                    for t in j.next:
                        lst.append(i.A[t].node)
                    print("当前节点的子节点列表：",lst)
                    print("------------------")
        if(max_delay[0]['class'] == 2):
            tree = 0
            for i in crossoffspring[max_delay[0]['num']].gi:
                print("################################")
                tree += 1
                print("修复树:",tree)
                for j in i.A:
                    print("------------------")
                    print("当前节点值：",j.node)
                    print("当前节点的父节点值：",i.A[j.parent].node)
                    lst = []
                    for t in j.next:
                        lst.append(i.A[t].node)
                    print("当前节点的子节点列表：",lst)
                    print("------------------")
        if(max_delay[0]['class'] == 3):
            tree = 0
            for i in mutationoffspring[max_delay[0]['num']].gi:
                print("################################")
                tree += 1
                print("修复树:",tree)
                for j in i.A:
                    print("------------------")
                    print("当前节点值：",j.node)
                    print("当前节点的父节点值：",i.A[j.parent].node)
                    lst = []
                    for t in j.next:
                        lst.append(i.A[t].node)
                    print("当前节点的子节点列表：",lst)
                    print("------------------")
        if(max_delay[0]['class'] == 4):
            tree = 0
            for i in localsearchoffspring[max_delay[0]['num']].gi:
                print("################################")
                tree += 1
                print("修复树:",tree)
                for j in i.A:
                    print("------------------")
                    print("当前节点值：",j.node)
                    print("当前节点的父节点值：",i.A[j.parent].node)
                    lst = []
                    for t in j.next:
                        lst.append(i.A[t].node)
                    print("当前节点的子节点列表：",lst)
                    print("------------------")
        
    

main()



