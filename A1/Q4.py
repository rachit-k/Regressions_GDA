import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import sys

#    (a)
ydata=[]
with open("q4x.dat") as f1:
    x = np.genfromtxt(f1)
for i in open("q4y.dat").readlines():
    ydata.append(i)
m=len(ydata)
y=np.zeros((m,1))
for i in range(m):
    if ydata[i]=='Canada\n':
        y[i]=1
        
x1=x[:,0]
x2=x[:,1]
x1=(x1-np.mean(x1))/np.std(x1)
x2=(x2-np.mean(x2))/np.std(x2)

x1=x1.reshape((m,1))
x2=x2.reshape((m,1))
#    x=np.concatenate((np.ones((m,1)),x1),axis=1)
x=np.concatenate((x1,x2),axis=1)
#    print(x)

sum1=0
mu1=np.array([0,0])
mu0=np.array([0,0])
x_00=[]
x_01=[]
x_10=[]
x_11=[]

for i in range(m):
    if y[i]==1:
        sum1=sum1+1
        mu1=mu1+x[i]
        x_11.append(x[i][1])
        x_10.append(x[i][0])
    else:
        mu0=mu0+x[i]
        x_01.append(x[i][1])
        x_00.append(x[i][0])
phi=sum1/m

sum0=m-sum1
mu1=mu1/sum1
mu0=mu0/sum0

mu1=mu1.reshape((1,2))
mu0=mu0.reshape((1,2))

sigma0=np.zeros((2,2))
sigma1=np.zeros((2,2))  
for i in range(m):
    if y[i]==1:
        sigma1=sigma1+np.dot((x[i]-mu1).T,(x[i]-mu1))
    else:
        sigma0=sigma0+np.dot((x[i]-mu0).T,(x[i]-mu0))

sigma0=sigma0/sum0
sigma1=sigma1/sum1

print("phi:")  
print(phi)
print("mu0:")  
print(mu0)
print("mu1:")  
print(mu1)
print("sigma0:")  
print(sigma0)
print("sigma1:")  
print(sigma1)

#    (b)-(f)

plt.plot(x_00,x_01,'o',label='Alaska',color='red',marker='x')
plt.plot(x_10,x_11,'o',label='Canada',color='blue',marker='+')
plt.xlabel('x1')
plt.ylabel('x2')

sigma=(sigma1*sum1+sigma0*sum0)/m
print("sigma:")  
print(sigma)
sigmainv=np.linalg.inv(sigma)
sigma0inv=np.linalg.inv(sigma0)
sigma1inv=np.linalg.inv(sigma1)

def linbd(x0,x1):
    x=np.array([x0,x1])
    return (1/2)*(np.dot((x-mu1),(np.dot(sigmainv,(x-mu1).T)))-np.dot((x-mu0),(np.dot(sigmainv,(x-mu0).T))))

def quadbd(x0,x1):
    x=np.array([x0,x1])
    return (1/2)*(np.dot((x-mu1),(np.dot(sigma1inv,(x-mu1).T)))-np.dot((x-mu0),(np.dot(sigma0inv,(x-mu0).T))))

linbd1=np.vectorize(linbd)
#    rx=linbound1()

rx=np.linspace(-3,3,100)
ry=np.linspace(-3,3,100)
rX,rY =np.meshgrid(rx,ry)
linbd1=np.vectorize(linbd)
z = linbd1(rX,rY)
z=z-(math.log((phi)/(1-phi)))

plt.contour(rX,rY,z,[0],colors='y',).collections[0].set_label('Linear Decision Boundary')

rx=np.linspace(-3,3,100)
ry=np.linspace(-3,3,100)
rX,rY =np.meshgrid(rx,ry)
quadbd1=np.vectorize(quadbd)
z = quadbd1(rX,rY)
z=z-(math.log((phi)/(1-phi)))- (1/2)*(math.log((np.linalg.det(sigma0))/(np.linalg.det(sigma1))))

plt.contour(rX,rY,z,[0],colors='g').collections[0].set_label('Quadratic Decision Boundary')
plt.legend()
plt.savefig("Q4.png",bbox_inches="tight")
plt.show()




