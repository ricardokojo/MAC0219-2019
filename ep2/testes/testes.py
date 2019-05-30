import statistics as st
import numpy as np
import matplotlib.pyplot as plt

NUMBER=['1','2','3','4','5','6','7','8','9','0']
time=''
minutes=float()
testes=[[],[]] #each vector inside matrix represents each time (real, user...)
with open ('pragma-2048-2.txt') as f:
    count =0
    for i in f:
        word=i.split(' ')
        if word[0] == 'real':
            for j in word[4]:
                if j in NUMBER or j=='.': 
                    time+=j
                if j == 'm': 
                    minutes=60*float(time)
                    time = ''
                if j == 's': 
                    testes[0].append(minutes+(float(time)))
                    time=''
        if word[0] == 'user': 
            for j in word[4]:
                if j in NUMBER or j=='.': 
                    time+=j
                if j == 'm': 
                    minutes=60*float(time)
                    time = ''
                if j == 's': 
                    testes[1].append(minutes+(float(time)))
                    time=''

            count+=1

'''
t=[[float (x.rstrip()[0:-1]) for x in i] for i in testes]
plt.plot(testes[0].sort())
plt.show()
#plt.hist(testes[1])
#plt.hist(testes[2])
'''
c=1.984 # 95% confidence
print('real_mean')
print(st.mean(testes[0]));
print('user_mean')
print(st.mean(testes[1]));
print('\n')
print('real_max')
print(st.mean(testes[0])+(st.stdev(testes[0]))*c/10);
print('real_min')
print(st.mean(testes[0])-(st.stdev(testes[0]))*c/10);
print('\n')
print('user_max')
print(st.mean(testes[1])+((st.stdev(testes[1])*c/10)));
print('user_min')
print(st.mean(testes[1])-((st.stdev(testes[1])*c/10)));

#plt.show()
