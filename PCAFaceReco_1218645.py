import sys
import numpy as np
from PCARow import PCARow
from PCACol import PCACol
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def main():
    pcah = PCARow()  # change to PCARow() to use row wise version of data
    
    # -------------------column wise version of PCA----------------
    [X, y, yseq] = pcah.read_images("C:/Users/ypava/OneDrive/Desktop/DataMining/ATTFaceDataSet/Testing")
    print(X[0].shape)
    [Xtest, ytest, yseqt] = pcah.read_images("C:/Users/ypava/OneDrive/Desktop/DataMining/ATTFaceDataSet/Training")
    
    # X is a list f 112x92 2-d arrays, y are the labels
    [E, EV, mu] = pcah.pca(pcah.asColumnMatrix(X), 100)  # top 100 Eigen vectors
    # E is the Eigen value array, EV is the catenated Eigen vectors, mu is the mean image
    print(EV.shape)
    
    # turn the first (at most) 16 eigenvectors into grayscale
    # images (note: eigenvectors are stored by column!)
    EF16 = []
    for i in range(min(len(X), 16)):
        e = EV[:, i].reshape(X[0].shape)
        EF16.append(pcah.normalize(e, 0, 255))  # for visualization purposes
    print(len(EF16))
    
    pcah.subplot(title="Eigenfaces AT&T Facedatabase", images=EF16, rows=4, cols=4,
                 sptitle=" Eigenface", colormap=cm.jet, filename="python_pca_eigenfaces.png")
    
    # reconstruct projections of first n Eigen Faces
    steps = [i for i in range(10, min(len(X), 200), 20)]
    EF10 = []
    for i in range(min(len(steps), 16)):
        numEvs = steps[i]
        P = pcah.project(EV[:, 0:numEvs], X[0].reshape(-1, 1), mu)
        R = pcah.reconstruct(EV[:, 0:numEvs], P, mu)
        # reshape and append to plots
        R = R.reshape(X[0].shape)
        EF10.append(pcah.normalize(R, 0, 255))
    # plot them and store the plot to "python_reconstruction.pdf"
    pcah.subplot(title="Reconstruction AT&T Facedatabase", images=EF10, rows=4, cols=4,
                 sptitle=" Eigenvectors", sptitles=steps, colormap=cm.gray,
                 filename=" python_pca_reconstruction.png")
    
    for xi in X:
        pcah.projections.append(pcah.project(EV, xi.reshape(-1, 1), mu))
    
    imtest = 20  # image number to test
    labelPredicted, index = pcah.predict(EV, Xtest[imtest], mu, y, yseq)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.asarray(Xtest[index]), cmap=cm.gray)
    plt.xlabel(ytest[25])
    plt.subplot(1, 2, 2)
    plt.imshow(np.asarray(Xtest[imtest]), cmap=cm.gray)
    plt.xlabel(labelPredicted)
    plt.show()
    print("actual label=" + ytest[imtest] + " label predicted=" + labelPredicted)
    
    # compute recognition accuracy
    i = 0
    accuracyCount = 0
    for xi in Xtest:
        labelPredicted, index = pcah.predict(EV, xi, mu, y, yseq)
        if labelPredicted == ytest[i]:
            accuracyCount = accuracyCount + 1
        i = i + 1
    print("recog accuracy = " + str(accuracyCount / i))

   

if __name__ == "__main__":
    sys.exit(int(main() or 0))
    

# import sys
# import numpy as np
# from PCARow import PCARow
# from PCACol import PCACol
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt

# def main():
#     pcah = PCARow()  # change to PCARow() to use row wise version of data
    
#     # Load training and testing images
#     [X, y, yseq] = pcah.read_images("C:/Users/ypava/OneDrive/Desktop/DataMining/ATTFaceDataSet/Testing")
#     [Xtest, ytest, yseqt] = pcah.read_images("C:/Users/ypava/OneDrive/Desktop/DataMining/ATTFaceDataSet/Training")

#     # Perform PCA on training data
#     [E, EV, mu] = pcah.pca(pcah.asRowMatrix(X), y)  # top 100 Eigen vectors

#     # Turn the first (at most) 16 eigenvectors into grayscale images
#     EF16 = []
#     for i in range(min(len(X), 16)):
#         e = EV[:, i].reshape(X[0].shape)
#         EF16.append(pcah.normalize(e, 0, 255))

#     # Plot the first 16 eigenfaces and store the plot
#     pcah.subplot(title="Eigenfaces AT&T Facedatabase", images=EF16, rows=4, cols=4,
#                  sptitle=" Eigenface", colormap=cm.jet, filename="python_pca_eigenfaces.png")

#     # Reconstruction steps
#     steps = [i for i in range(10, min(len(X), 200), 20)]
#     EF10 = []
#     for i in range(min(len(steps), 16)):
#         numEvs = steps[i]
#         P = pcah.project(EV[:, 0:numEvs], X[0].reshape(1, -1), mu)
#         R = pcah.reconstruct(EV[:, 0:numEvs], P, mu)
#         # Reshape and append to plots
#         R = R.reshape(X[0].shape)
#         EF10.append(pcah.normalize(R, 0, 255))

#     # Plot reconstructed images and store the plot
#     pcah.subplot(title="Reconstruction AT&T Facedatabase", images=EF10, rows=4, cols=4,
#                  sptitle=" Eigenvectors", sptitles=steps, colormap=cm.gray,
#                  filename="python_pca_reconstruction.png")

#     # Project training images onto eigenfaces
#     for xi in X:
#         pcah.projections.append(pcah.project(EV, xi.reshape(1, -1), mu))

#     # Predict the label for the 25th test image and visualize it
#     labelPredicted, index = pcah.predict(EV, Xtest[25], mu, y, yseq)
#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.imshow(np.asarray(Xtest[index]), cmap=cm.gray)
#     plt.xlabel(ytest[25])
#     plt.subplot(1, 2, 2)
#     plt.imshow(np.asarray(Xtest[25]), cmap=cm.gray)
#     plt.xlabel(labelPredicted)
#     plt.show()
#     print("actual label=" + ytest[25] + " label predicted=" + labelPredicted)

#     # Compute recognition accuracy
#     i = 0
#     accuracyCount = 0
#     for xi in Xtest:
#         labelPredicted, index = pcah.predict(EV, xi, mu, y, yseq)
#         if labelPredicted == ytest[i]:
#             accuracyCount = accuracyCount + 1
#         i = i + 1
#     print("recog accuracy = " + str(accuracyCount / i))


# if __name__ == "__main__":
#     sys.exit(int(main() or 0))


