import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# we create two clusters of random points
n_samples_1 = 1000
n_samples_2 = 100
centers=[[0,0],[2,2]]
std=[1.5,0.5]

X,y= make_blobs(n_samples=[n_samples_1,n_samples_2], centers=centers, 
                cluster_std=std, random_state=0, shuffle=False)

# without class weights
clf= svm.SVC(kernel='linear')
clf.fit(X,y)

plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap='Paired',edgecolors='k')
# NOTE!!!: Ensure to plot the scatter plot first before getting plt.gca()

ax=plt.gca()
xlim=ax.get_xlim()
ylim=ax.get_ylim()

xx=np.linspace(xlim[0],xlim[1],30)
yy=np.linspace(ylim[0],ylim[1],30)

XX, YY=np.meshgrid(xx,yy)
Z=clf.decision_function(np.column_stack([XX.ravel(),YY.ravel()]))
Z=Z.reshape(XX.shape)

a=plt.contour(xx,yy, Z, alpha=0.5, levels=[0], linestyles=['-'], colors='k')
# we assigned contour plot to 'a' so we can put legend

# with class weights
wclf = svm.SVC(kernel='linear', class_weight={1: 3})
wclf.fit(X, y)

Z = wclf.decision_function(np.column_stack([XX.ravel(),YY.ravel()])).reshape(XX.shape)
b=plt.contour(XX, YY, Z, colors='r', levels=[0], alpha=0.5, linestyles=['-'])
# we assigned contour plot to 'b' so we can put legend

#Put labels/legend
plt.legend([a.collections[0],b.collections[0]],['non weighted','weighted'],loc='lower left')
plt.title('Class weight 1:3')


