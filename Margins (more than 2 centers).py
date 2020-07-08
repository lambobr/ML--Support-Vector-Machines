import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets

# Create dataset
X,y = datasets.make_blobs(n_samples=150, centers=3, random_state=0)

# Train model
model = svm.SVC(kernel = 'linear')
model.fit(X,y)

# Plot the decision boundary

    # alternate way of getting the limits
xlim = [X[:,0].min(),X[:,0].max()]
ylim = [X[:,1].min(),X[:,1].max()]

    # we use very small step size so that the boundary is smooth
    # we use np.arange instead of np.linspace
xx = np.arange(xlim[0]-1,xlim[1]+1,.01) 
yy = np.arange(ylim[0]-1,ylim[1]+1,.01)

XX, YY = np.meshgrid(xx,yy)

    # we use predict instead of decision_function
Z = model.predict(np.column_stack([XX.ravel(),YY.ravel()]))
Z = Z.reshape(XX.shape)
    
    # surface contour (contourf)
plt.contourf(XX, YY, Z, cmap='viridis', alpha=0.8)

# Plot the data
    # we plot the data last so it is in front of the surface contour plot
plt.scatter(X[:,0],X[:,1],c=y,cmap='Paired',edgecolors='k')

            
            
            