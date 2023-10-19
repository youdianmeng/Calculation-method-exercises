"""
Created by Lucio on 2023/3/2
python 3.6.15
隶属度函数
"""
import numpy as np
import math
import matplotlib.pyplot as plt


def mu_x(a):
    list1 =[]
    list1.append(a)
    list1.append(2 - a) 
    return list1

def mu_y(a): 
    list1 =[]
    list1.append(1 + a)
    list1.append(3 - a) 
    return list1 

def sinxcosx(list1, list2):
    lst = []
    for i in list1:
        for j in list2:
            lst.append(math.sin(i+j)*math.cos(i*j)+i*j*j -i*i*i)
    return min(lst), max(lst)

def plot_data(l1,l2):
    z1 = np.polyfit(l1, l2, 20)
    yvals = np.polyval(z1, l1)
    plt.plot(l1, l2,'*',label='original values')
    plt.plot(l1, yvals,'r',label='polyfit values')
    plt.title("Membership Function")
    #plt.xlabel('x, y')
    #plt.ylabel('μ(x,y)')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.show()
    
def main():
    a = list(np.random.rand(50))
    a.sort()
    x = []
    y = []
    for i in a:
        y.append(i)
        y.append(i)
        list1 = mu_x(i)
        list2 = mu_y(i)
        minn, maxx = sinxcosx(list1, list2)
        x.append(minn)
        x.append(maxx)
    #xy = sorted(zip(x,y))
    #print(xy)
    l1, l2 = (list(t) for t in zip(*sorted(zip(x, y))))
    plot_data(l1,l2)
    return 0  


if __name__ == '__main__':
    print(main())
    