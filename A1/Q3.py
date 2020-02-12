import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import sys


def main():
#    (a)
    with open("logisticX.csv") as f1:
        x = np.genfromtxt(f1,delimiter=',')
    with open("logisticY.csv") as f2:
        y = np.genfromtxt(f2,delimiter=',') 

    x1=x[:,0]
    x2=x[:,1]
    x1=(x1-np.mean(x1))/np.std(x1)
    x2=(x2-np.mean(x2))/np.std(x2)
    
    m=y.size

    x1=x1.reshape((m,1))
    x2=x2.reshape((m,1))
    x=np.concatenate((np.ones((m,1)),x1),axis=1)
    x=np.concatenate((x,x2),axis=1)#    m*3
    y=y.reshape((100,1))
    theta=np.zeros((3,1))

    k=0
    lthetaold=-1000000
    while True:
        hthetax=np.reciprocal(np.ones((m,1)) + np.exp((-1.0)*np.matmul(x,theta)))

        delltheta=(np.matmul((y-hthetax).T,x)).T

        D=np.zeros((m,m))
        for i in range(m):
            D[i][i]=(hthetax[i][0])*(1-hthetax[i][0])

        H=np.matmul(np.transpose(x),np.matmul(D,x))

        hinv=np.linalg.inv(H) #3*3

        theta=theta-np.matmul(hinv,delltheta)
        theta=theta/np.linalg.norm(theta)

        ltheta0=0
        ltheta1=0
        for i in range(m):
            if y[i]==0:
                ltheta0= ltheta0+np.sum(np.log(np.ones((m,1))-hthetax[i]))
            else:
                ltheta1=ltheta1+np.sum(np.log(hthetax[i]))       
        ltheta=ltheta0+ltheta1

        k=k+1
        if (abs(ltheta-lthetaold))<0.000001:
            break
        lthetaold=ltheta
    
#    (b)
    print('no of iterations:')
    print(k)
    x11=[]
    x22=[]
    y11=[]
    y22=[]
    print("theta:")  
    print(theta)
    for i in range(m):
        if y[i]==0.0:
            x11.append(x[i][1])
            x22.append(x[i][2])
        else:
            y11.append(x[i][1])
            y22.append(x[i][2])
            
    plt.plot(x11,x22,'o',color='red',label='0',marker='x')
    plt.plot(y11,y22,'o',color='blue',label='1',marker='+')
    plt.plot(-1*x1,(theta[0]/theta[1]) + (theta[2]/theta[1])*x1,label='Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
#    plt.savefig("Q3.png",bbox_inches="tight")
    plt.show()
    
    
main()    

