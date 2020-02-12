import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import sys
from mpl_toolkits.mplot3d import Axes3D

with open("linearX.csv") as f1:
    x = np.genfromtxt(f1)
with open("linearY.csv") as f2:
    y = np.genfromtxt(f2)
x=(x-np.mean(x))/np.std(x)

rate=0.001

m=x.size
theta0=0
theta1=0
theta0_list=[]
theta1_list=[]
errors=[]

e=0
k=0
while True:
    eold=e
    deltheta0=np.sum(y-theta0-theta1*x)/m
    deltheta1=np.sum(np.multiply((y-theta0-theta1*x),x))/m
    
    theta0=theta0+rate*deltheta0
    theta1=theta1+rate*deltheta1
    theta0_list.append(theta0)
    theta1_list.append(theta1)
    
    e=np.sum((y-theta0-theta1*x)**2)/(2*m)
    errors.append(e)
    k=k+1
    if (abs(e-eold))<0.0000000001:
        break
print('no of iterations:')
print(k)
print('values:')    
print(theta0)
print(theta1)
#    (b)    
plt.plot(x,y,'o',label='data')
plt.plot(x,theta0+theta1*x,label='Hypothesis Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.savefig("Q1.png",bbox_inches="tight")
plt.show()


#    (c)


fig = plt.figure()

ax = fig.gca(projection='3d')
ax.set_xlim3d([0.0, 1.0])
ax.set_ylim3d([0.0, 0.002])   
ax.set_zlim3d([0.0, 1.0])
ax.set_xlabel('thetha0')
ax.set_ylabel('thetha1')
ax.set_zlabel('errors')    
#    ax.set_title('3D Test')

ax.scatter(theta0_list,theta1_list,errors,label='points')
plt.legend()
#plt.savefig("Q1-2.png",bbox_inches="tight")
plt.show()

def fx(rx, ry):
    f=0.0
    for i in range(y.size):
        f=f+(y[i]-ry-x[i]*rx)**2
    return f/200

rx=np.linspace(-3,3,100)
ry=np.linspace(-3,3,100)
rX,rY =np.meshgrid(rx,ry)

rZ=fx(rX,rY)
fig.clf()
fig = plt.figure()
ax = fig.gca(projection='3d') 

ax.plot_surface(rX, rY, rZ, linewidth=0,  cmap='Reds_r', alpha=0.2)
#
#for i in range(k):
#    ax.scatter3D(theta0_list[k-i-1], theta1_list[k-i-1],errors[k-i-1] , color='green')
#     plt.pause(0.2)
#plt.show()

#    (d)
fig.clf()
fig = plt.figure() 

for i in range(k):
    plt.contour(rX, rY, rZ, [errors[i]])
#    plt.pause(0.1)
plt.show()


    
