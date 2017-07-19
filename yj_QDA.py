# Classification basics

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

# Quadratic Discriminant Analysis

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
model = QuadraticDiscriminantAnalysis().fit(X, y)

x1 = [[0, 0]]
x2 = [[0.5, -1]]
x3 = [[0.1, 2]]
p1 = model.predict_proba(x1)[0]
p2 = model.predict_proba(x2)[0]
p3 = model.predict_proba(x3)[0]

plt.subplot(411)
plt.scatter(X.T[0], X.T[1], c=y, s=100, cmap=mpl.cm.brg)
plt.scatter(x1[0][0], x1[0][1], c='r', s=100)
plt.scatter(x2[0][0], x2[0][1], c='black', s=100)
plt.scatter(x3[0][0], x3[0][1], c='yellow', s=100)
plt.title("data")
plt.subplot(412)
plt.bar(model.classes_, p1, align="center")
plt.title("conditional probability Red")
plt.axis([0, 3, 0, 1])
plt.xticks(model.classes_)
plt.subplot(413)
plt.bar(model.classes_, p2, align="center")
plt.title("conditional probability Blue")
plt.axis([0, 3, 0, 1])
plt.subplot(414)
plt.bar(model.classes_, p3, align="center")
plt.title("conditional probability Yellow")
plt.axis([0, 3, 0, 1])
plt.gca().xaxis.grid(False)
plt.xticks(model.classes_)
plt.tight_layout()
plt.show()