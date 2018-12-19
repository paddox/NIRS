import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pickle
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import Normalize

class QRS:
    def __init__(self, px, py, qx, qy, rx, ry, sx, sy, tx, ty):
        self.px = px
        self.py = py
        self.qx = qx
        self.qy = qy
        self.rx = rx
        self.ry = ry
        self.sx = sx
        self.sy = sy
        self.tx = tx
        self.ty = ty
    def printconsole(self):
        print("px:", self.px)
        print("py:", self.py)
        print("qx:", self.qx)
        print("qy:", self.qy)
        print("rx:", self.rx)
        print("ry:", self.ry)
        print("sx:", self.sx)
        print("sy:", self.sy)
        print("tx:", self.tx)
        print("ty:", self.ty)

def learning(X, y):
    '''
    # импортируем набор данных (например, возьмём тот же iris)
    iris = datasets.load_iris()
    X = iris.data[:, :2] # возьмём только первые 2 признака, чтобы проще воспринять вывод
    y = iris.target
    '''


    C = 1.0 # параметр регуляризации SVM
    svc = svm.SVC(kernel='rbf', C=100,gamma=10000).fit(X, y) # здесь мы взяли линейный kernel

    # создаём сетку для построения графика
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    h = 0.001

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))

    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length') # оси и название укажем на английском
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('SVC with linear kernel')
    plt.show()

def logregression(X, y):
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

def baies(X, y):
    from sklearn import metrics
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    model = GaussianNB()
    model.fit(X, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

def knearest(X, y):
    from sklearn import metrics
    from sklearn.neighbors import KNeighborsClassifier
    '''
    k_range = list(range(1,16))
    print("param_grid")
    param_grid = dict(n_neighbors=k_range)
    print("cv")
    cv = StratifiedShuffleSplit()
    print("grid")
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)
    print("The best parameters are {0} with a score of {1:.4f}".format(grid.best_params_, grid.best_score_))
    print(grid.cv_results_['mean_test_score'])

    plt.bar(k_range, grid.cv_results_['mean_test_score'])
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.show()
    '''
    
    # fit a k-nearest neighbor model to the data
    model = KNeighborsClassifier()
    model.fit(X, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    

def cart(X, y):
    from sklearn import metrics
    from sklearn.tree import DecisionTreeClassifier
    # fit a CART model to the data
    model = DecisionTreeClassifier()
    model.fit(X, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

from matplotlib.colors import Normalize
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def supportvm(X, y):
    # fit a SVM model to the data
    '''
    C_range = np.logspace(0, 2, 10)
    gamma_range = np.logspace(1, 3, 10)
    print("param_grid")
    param_grid = dict(gamma=gamma_range, C=C_range)
    print("cv")
    cv = StratifiedShuffleSplit()
    print("grid")
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
    grid.fit(X, y)
    print("The best parameters are {0} with a score of {1:.2f}".format(grid.best_params_, grid.best_score_))
    #Draw color map
    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
            norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()
    '''
    
    model = SVC(gamma=100, C=100)
    model.fit(X, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    
    

def histogram(data, n_bins, cumulative=False, x_label = "", y_label = "", title = ""):
    _, ax = plt.subplots()
    ax.hist(data, bins = n_bins, cumulative = cumulative, color = '#539caf')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    plt.show()

def overlaid_histogram(data1, data2, n_bins = 0, data1_name="", data1_color="#539caf", data2_name="", data2_color="#7663b0", x_label="", y_label="", title=""):
    # Set the bounds for the bins so that the two distributions are fairly compared
    max_nbins = 10
    data_range = [min(min(data1), min(data2)), max(max(data1), max(data2))]
    binwidth = (data_range[1] - data_range[0]) / max_nbins


    if n_bins == 0:
    	bins = np.arange(data_range[0], data_range[1] + binwidth, binwidth)
    else: 
    	bins = n_bins

    # Create the plot
    _, ax = plt.subplots()
    ax.hist(data1, bins = bins, color = data1_color, alpha = 1, label = data1_name)
    ax.hist(data2, bins = bins, color = data2_color, alpha = 0.75, label = data2_name)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc = 'best')
    plt.show()

#export from file
with open('ecgiddb_qrs.dat', 'rb') as f:
    data_new = pickle.load(f)
    X = []
    y = []
    for i in range(len(data_new)):
        for j in range(min(len(data_new[i].py), 
                           len(data_new[i].qy),
                           len(data_new[i].ry),
                           len(data_new[i].sy),
                           len(data_new[i].ty))-1):
            '''
            X.append([data_new[i].py[j], data_new[i].py[j+1], data_new[i].py[j+2],
                      data_new[i].qy[j], data_new[i].qy[j+1], data_new[i].qy[j+2],
                      data_new[i].ry[j], data_new[i].ry[j+1], data_new[i].ry[j+2],
                      data_new[i].sy[j], data_new[i].sy[j+1], data_new[i].sy[j+2],
                      data_new[i].ty[j], data_new[i].ty[j+1], data_new[i].ty[j+2]])
                    '''  
            
            X.append([data_new[i].py[j], data_new[i].py[j+1],
                      data_new[i].qy[j], data_new[i].qy[j+1],
                      data_new[i].ry[j], data_new[i].ry[j+1],
                      data_new[i].sy[j], data_new[i].sy[j+1],
                      data_new[i].ty[j], data_new[i].ty[j+1]])
            '''
            X.append([data_new[i].py[j],
                      data_new[i].qy[j],
                      data_new[i].ry[j],
                      data_new[i].sy[j],
                      data_new[i].ty[j]])
                     
'''

            y.append(1 if i % 2 == 0 else 0)
    X = np.array(X)
    y = np.array(y)
    print(X, y)
    #learning(X, y)
    #logregression(X, y)
    baies(X, y)
    #knearest(X, y)
    #cart(X, y)
    #supportvm(X, y)
    
    