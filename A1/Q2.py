import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import sys


def main():
#    (a)
    theta_initial=np.array([3,1,2])
    x=np.ones((1000000,1))
    x1=np.random.normal(3,2,1000000)
    x1=x1.reshape((1000000,1))
    x2=np.random.normal(-1,2,1000000)
    x2=x2.reshape((1000000,1)) 
    e=np.random.normal(0,2**0.5,1000000)
    x=np.concatenate((x,x1),axis=1)
    x=np.concatenate((x,x2),axis=1)
    y=np.dot(x,theta_initial)+e
    
#    (b)
    theta=np.array([0,0,0])
    rate=0.001
    m=1000000
    r=100
    errors=[]
    theta0_list=[]
    theta1_list=[]
    theta2_list=[]
    k=0
    j=0
    e=0
    while True: 
        deltheta=0
        e=0
        for i in range(r):
            deltheta=deltheta+(np.dot(np.sum(y[int(i+k*r)]-np.dot(x[int(i+k*r)],theta)),x[int(i+k*r)]))/(2*r)
        theta=theta+rate*deltheta
        theta0_list.append(theta[0])
        theta1_list.append(theta[1])
        theta2_list.append(theta[2])
        for i in range(r):
            e=e + ((y[int(i+k*r)]-np.dot(x[int(i+k*r)],theta))**2)
        e=e/(2*r)
        k=k+1
#        errors.append(e)

#        eavg=0
#        if(len(errors)>1000):
#            for i in range(1000):
#                eavg=eavg+errors[len(errors)-i-1]
#            eavg=eavg/1000
#        if abs(e-eavg)<0.000000000001:
#            break
        if k>=(m/r):
            k=0
            j=j+1
            print(j)
#        if j>=1 or k>20000: #1            
        if j>=3:  #100
#        if j>=250:  #10000
            break
    print("epoch:")           
    print(j)     
    print("theta:")           
    print(theta)
    print("error:")
    print(e)  
    
#    (c)
    with open("q2test.csv") as f1:
        rd = np.genfromtxt(f1,delimiter=',')
        
    x1=rd[1:,0]
    x2=rd[1:,1]
    y=rd[1:,2]
    n=x1.size
    x1=x1.reshape((n,1))
    x2=x2.reshape((n,1))
    x=np.concatenate((np.ones((n,1)),x1),axis=1)
    x=np.concatenate((x,x2),axis=1)
#    print(x.shape)
    e=0
    for i in range(n):
        e=e + ((y[i]-np.dot(x[i],theta))**2)
    e=e/(2*n)
    print("error on test data:")
    print(e)
    
#    (d)
    ax = plt.axes(projection='3d')
    ax.set_xlim3d([0.0, 3.0])
    ax.set_ylim3d([0.0, 3.0])   
    ax.set_zlim3d([0.0, 3.0])
    ax.set_xlabel('thetha0')
    ax.set_ylabel('thetha1')
    ax.set_zlabel('thetha2')    
    
    ax.scatter(theta0_list,theta1_list,theta2_list,label='points')
    plt.legend()
#    plt.savefig("Q2-1000000.png",bbox_inches="tight")
    plt.show()
    
    
main()    




