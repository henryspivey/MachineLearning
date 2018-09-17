import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
data= load_iris()

features = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names


labels = target_names[target]

fig, axes = plt.subplots(2,3)
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

color_markers = [
    ('r', '>'),
    ('g', 'o'),
    ('b', 'x')
]

for i, (p0, p1) in enumerate(pairs):
    ax = axes.flat[i]
    for t in range(3):
        c,marker = color_markers[t]
        ax.scatter(features[target == t, p0], features[target == t, p1], marker=marker, c=c)
    ax.set_xlabel(feature_names[p0])
    ax.set_ylabel(feature_names[p1])
    ax.set_xticks([])
    ax.set_yticks([])

# fig.tight_layout()
# fig.show()

from sklearn import tree
tr = tree.DecisionTreeClassifier(min_samples_leaf=10)
tr.fit(features, labels)

import graphviz
tree.export_graphviz(tr, feature_names=feature_names, rounded=True, out_file='decision.dot')
# graphviz.Source(open('decision.dot').read())
graphviz.render('dot',format='png', filepath='decision.dot')

prediction = tr.predict(features)
print("Accuracy: {:.1%}".format(np.mean(prediction == labels)))

# cross validation - take an example out of the training data
predictions = []
for i in range(len(features)):
    train_features= np.delete(features, i, axis=0)
    train_labels = np.delete(labels, i, axis=0)
    tr.fit(train_features, train_labels)
    predictions.append(tr.predict([features[i]]))
predictions = np.array(predictions)

# cross validation with sklearn
from sklearn import model_selection
predictions= model_selection.cross_val_predict(
    tr,
    features,
    labels,
    cv = model_selection.LeaveOneOut()
)
print(np.mean(predictions==labels))

# nearest neighbor classification
#  When classifying a new element, this looks at the training data. For the object that is closest to it, its nearest neighbor. Then, it returns its label as the answer.

import load
from load import load_dataset
feature_names = [
    'area',
    'perimeter',
    'compactness',
    'length of kernel',
    'width of kernel',
    'asymmetry coefficient',
    'length of kernel groove',
]

data = load_dataset('seeds')
features = data['features']
target = data['target']


from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
knn = KNeighborsClassifier(n_neighbors=1)

kf = model_selection.KFold(n_splits=5, shuffle=False)
means = []
for training, testing in kf.split(features):
    knn.fit(features[training], target[training])
    prediction = knn.predict(features[testing])

    curmean = np.mean(prediction  == target[testing])
    means.append(curmean)
print('Mean Accuracy: {:.1%}'.format(np.mean(means)))


#normalization
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

clf  = KNeighborsClassifier(n_neighbors=1)
clf = Pipeline([('norm', StandardScaler()) , ('knn', clf)])

# random forest
# based on decision trees
from sklearn import  ensemble
rf = ensemble.RandomForestClassifier(n_estimators=100)
predict = model_selection.cross_val_predict(rf, features, target)
print("RF accuracy: {:.1%}".format(np.mean(predict == target)))
