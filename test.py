'''
Description: test
Author: Xu Jiaming
Date: 2022-04-28 08:04:38
LastEditTime: 2022-04-28 17:59:15
LastEditors:  
FilePath: test.py
'''
lst = [73, 74, 75, 71, 69, 72, 76, 73]
lst_minu=[]
lst_minu_sum = []
ans = []
for i in range(len(lst)):
    pd = 1
    for j in range(i+1,len(lst)):
        if(lst[j] > lst[i]):
            pd = 0
            ans.append(j-i)
            break
    if(pd):
        ans.append(0)
print(ans)


