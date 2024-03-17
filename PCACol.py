import numpy as np
import os
import sys
from PIL import Image  
import matplotlib.cm as cm  
import matplotlib.pyplot as plt
from DistanceMetric import *


class PCACol(object):  # column wise data computations
    def __init__(self):
        self.projections = []
        self.dist_metric = EuclideanDistance()

    def asRowMatrix(self, X):
        if len(X) == 0:
            return np.array([])
        mat = np.empty((0, X[0].size), dtype=X[0].dtype)
        for row in X:
            mat = np.vstack((mat, np.asarray(row).reshape(1, -1)))
        return mat

    def asColumnMatrix(self, X):
        if len(X) == 0:
            return np.array([])
        mat = np.empty((X[0].size, 0), dtype=X[0].dtype)
        for col in X:
            mat = np.hstack((mat, np.asarray(col).reshape(-1, 1)))
        return mat

    def pca(self, X, num_components=0):
        [d, n] = X.shape  # n - number of images, d = input dimension e.g. 112x92=10304
        if (num_components <= 0) or (num_components > n):
            num_components = n
        mu = X.mean(axis=1)
        X = (X.T - mu).T
        if d < n:
            Cov = np.dot(X, X.T)  # covariance matrix
            [eigenvalues, EV] = np.linalg.eigh(Cov)
        else:
            Cov = np.dot(X.T, X)
            [eigenvalues, eigenvectors] = np.linalg.eigh(Cov)
        EV = np.dot(X, eigenvectors)  # it is termed as EigenFace
        for i in range(n):
            EV[:, i] = EV[:, i] / np.linalg.norm(EV[:, i])
        idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx]
        EV = EV[:, idx]
        eigenvalues = eigenvalues[0:num_components].copy()
        EV = EV[:, 0:num_components].copy()
        return [eigenvalues, EV, mu]

    def normalize(self, X, low, high, dtype=None):
        X = np.asarray(X)
        minX, maxX = np.min(X), np.max(X)
        X = (X - minX) / (maxX - minX) * (high - low)
        X = X + low
        if dtype is None:
            return np.asarray(X)
        return np.asarray(X, dtype=dtype)

    def read_images(self, path, sz=None):
        X, y, yseq = [], [], []  # list of images and labels, yseq is sequential ordering number
        for dirname, dirnames, filenames in os.walk(path):
            for filename in filenames:
                try:
                    fname = os.path.join(path, filename)
                    im = Image.open(fname)
                    im = im.convert("L")
                    if sz is not None:
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    label = filename.split('_', 1)[0]
                    y.append(label)
                    yseq.append(len(X))
                except IOError:
                    print(f"I/O error: {sys.exc_info()[0]}")
        return X, y, yseq

    def project(self, EF, X, mu=None):
        if mu is None:
            return np.dot(EF.T, X)
        return np.dot(EF.T, (X.T - mu).T).T

    def reconstruct(self, EF, P, mu=None):
        if mu is None:
            return np.dot(EF, P.T)
        return (np.dot(EF, P.T).T + mu).T

    def create_font(self, fontname='Tahoma', fontsize=10):
        return {'fontname': fontname, 'fontsize': fontsize}

    def subplot(self, title, images, rows, cols, sptitle="subplot", sptitles=[], colormap=cm.gray, ticks_visible=True,
                filename=None):
        fig = plt.figure()
        fig.text(.5, .95, title, horizontalalignment='center')
        for i in range(len(images)):
            ax0 = fig.add_subplot(rows, cols, (i + 1))
            plt.setp(ax0.get_xticklabels(), visible=False)
            plt.setp(ax0.get_yticklabels(), visible=False)
            if len(sptitles) == len(images):
                plt.title(f"{sptitle} #{sptitles[i]}", self.create_font('Tahoma', 10))
            else:
                plt.title(f"{sptitle} #{(i + 1)}", self.create_font('Tahoma', 10))
            plt.imshow(np.asarray(images[i]), cmap=colormap)
        if filename is None:
            plt.show()
        else:
            fig.savefig(filename)

    def predict(self, EF, X, mu, y, yseq):
        minDist = np.finfo('float').max
        minClass = -1
        index = -1
        Q = self.project(EF, X.reshape(-1, 1), mu).T
        for i in range(len(self.projections)):
            dist = self.dist_metric(self.projections[i], Q)
            if dist < minDist:
                minDist = dist
                minClass = y[i]
                index = yseq[i]
        return minClass, index
