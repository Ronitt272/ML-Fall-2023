import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import pickle

hw4reg = pickle.load(open('hw4reg.pkl', 'rb'))
x0,y0=hw4reg['data'],hw4reg['labels']

#Problem 3 with KERNEL

def kern_func(xi,zi):
    #print(xi)
    kern_val = min(xi,zi)
    return kern_val

def kernel_mat(xs):
    kern_mat=[]
    for x1 in xs:
        kern_mat_row = []
        for x2 in xs:
            kern_mat_row.append(kern_func(x1,x2))
        kern_mat.append(kern_mat_row)
    return kern_mat

def kernel_vec(xs,cont_x):
    kern_vec=[]
    for xi in xs:
        kern_vec.append(kern_func(xi,cont_x))
    return kern_vec
    
def opt_y(x_var,x0s,y0s,l,km0):
    #print(x0s[0])
    wTphi = np.matmul(y0s,np.matmul(km0,kernel_vec(x0s,x_var)))
    
    return wTphi

#km=npl.inv(kernel_mat(x0)+l*np.identity(len(y0),dtype="float"))
kms=[]
ls=[]
for i in range(-20,21):
    l=2**i
    ls.append(i)
    kms.append(npl.inv(kernel_mat(x0)+l*np.identity(len(y0),dtype="float")))

print("DONE")

def mean_squared_error(y1s,y2s):
    mse=0
    for i in range(0,len(y1s)):
        mse+=(1/len(y1s))*((y1s[i]-y2s[i])**2)
    return mse
mserrs=[]
mins=[1000000,0]
x01,y01=hw4reg['valdata'], hw4reg['vallabels']
print(len(x01),len(y01))
for i in range(0,len(kms)):
    y_preds=[]
    for xi in x01:
        y_preds.append(opt_y(xi,x0,y0,i,kms[i]))
    mserrs.append(mean_squared_error(y01,y_preds))
    mserr=mean_squared_error(y01,y_preds)
    if mserr<mins[0]:
        mins[0]=mean_squared_error(y01,y_preds)
        mins[1]=i-20 #subract 20 becase that is the offset from the index
print("mins are",mins)
print(mserrs)

plt.figure()
plt.scatter(ls, mserrs)
#plt.plot(hw4reg['testdata'], hw4reg['testlabels'], '.',color="orange")
#plt.plot(xp, yp)
#plt.plot(hw4reg['grid'], pred, linestyle='-')
#plt.legend(['training data', '$f(x)$'], loc='lower left')
plt.xlabel('log2(lambda)')
plt.ylabel('MSE')
plt.savefig('hw4lambdas.pdf', bbox_inches='tight')
    
    #print(i)


xp = np.linspace(0,1,1000)
yp=[]
for xi in xp:
    yp.append(opt_y(xi,x0,y0,0,kms[18]))

    
print(len(yp),len(xp))
plt.figure()
plt.plot(hw4reg['data'], hw4reg['labels'], '.')
#plt.plot(hw4reg['testdata'], hw4reg['testlabels'], '.',color="orange")
plt.plot(xp, yp)
#plt.plot(hw4reg['grid'], pred, linestyle='-')
plt.legend(['training data', '$f(x)$'], loc='lower left')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('hw4reg.pdf', bbox_inches='tight')