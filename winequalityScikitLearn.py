import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtreeviz.trees import dtreeviz  # used for tree visualization

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, tree
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.cluster import KMeans, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def linearRegression():

    # just to fit the decision tree in a picture
    model = RandomForestRegressor(max_depth=3)
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)

    mse = metrics.mean_squared_error(Y_test, Y_predict)
    print("\tMean Squared Error:", mse)

    mae = metrics.mean_absolute_error(Y_test, Y_predict)
    print("\tMean Absolute Error:", mae)

    mape = metrics.mean_absolute_percentage_error(Y_test, Y_predict)
    print("\tMean Absolute Percentage Error:", mape)

    mdae = metrics.median_absolute_error(Y_test, Y_predict)
    print("\tMedian Absolute Error:", mdae)

    plt.figure(figsize=(20, 20))
    _ = tree.plot_tree(model.estimators_[
                       0], feature_names=x.columns, filled=True)
    plt.show()

    viz = dtreeviz(model.estimators_[0], X_train, y,
                   feature_names=X_train.columns, target_name="Target")
    viz.view()


def classification():

    model = RandomForestClassifier(max_depth=3)
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)

    cm = metrics.confusion_matrix(Y_test, Y_predict)
    print("Confusion Matrix:")
    print(cm)

    prfs = metrics.precision_recall_fscore_support(Y_test, Y_predict)
    print("Precision Recall F-score Support:")
    print(prfs)

    accuracy = metrics.accuracy_score(Y_test, Y_predict)
    print("Accuracy:")
    print(accuracy)

    cr = metrics.classification_report(Y_test, Y_predict)
    print("Classification Report:")
    print(cr)

    plt.figure(figsize=(20, 20))
    _ = tree.plot_tree(model.estimators_[
                       0], feature_names=x.columns, filled=True)
    plt.show()

    viz = dtreeviz(model.estimators_[1], X_train, y,
                   feature_names=X_train.columns, target_name="Target")
    viz.view()


def clustering(showPlots):
    # function that returns a figure for each feature clustering by quality

    y_pred_kmeans = KMeans(n_clusters=10, random_state=1).fit_predict(x)
    y_pred_meanshift = MeanShift().fit_predict(x)
    y_pred_gaussianmixture = GaussianMixture(n_components=10).fit_predict(x)

    ssKMeans = metrics.silhouette_score(x, y_pred_kmeans)
    ssMeanShift = metrics.silhouette_score(x, y_pred_meanshift)
    ssGaussianMixture = metrics.silhouette_score(x, y_pred_gaussianmixture)

    print("Shilhouette Score using KMeans cluster is: ", ssKMeans)
    print("Shilhouette Score using MeanShifth cluster is: ", ssMeanShift)
    print("Shilhouette Score using GaussianMixture cluster is: ", ssGaussianMixture)

    if showPlots:
        for col in headers[:-1]:

            fig, axs = plt.subplots(2, 2)
            fig.suptitle("clustering of " + col +
                         " and " + headers[-1], fontsize=16)

            axs[0, 0].scatter(x[headers[headers.index(col)]], y, c=y)
            axs[0, 0].set_title("Groundtruth Data")
            axs[0, 0].set_xlabel(col)
            axs[0, 0].set_ylabel(headers[-1])

            axs[0, 1].scatter(x[headers[headers.index(col)]],
                              y, c=y_pred_kmeans)
            axs[0, 1].set_title("KMeans")
            axs[0, 1].set_xlabel(col)
            axs[0, 1].set_ylabel(headers[-1])

            axs[1, 0].scatter(x[headers[headers.index(col)]],
                              y, c=y_pred_meanshift)
            axs[1, 0].set_title("MeanShift")
            axs[1, 0].set_xlabel(col)
            axs[1, 0].set_ylabel(headers[-1])

            axs[1, 1].scatter(x[headers[headers.index(col)]],
                              y, c=y_pred_gaussianmixture)
            axs[1, 1].set_title("Gaussian Mixture")
            axs[1, 1].set_xlabel(col)
            axs[1, 1].set_ylabel(headers[-1])

            plt.show()
            plt.close(fig)


def dimensionalityReduction():

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.scatter(x[headers[0]], x[headers[1]], x[headers[2]], c=y)
    ax.set_xlabel(headers[0])
    ax.set_ylabel(headers[1])
    ax.set_zlabel(headers[2])

    pca = PCA(n_components=2)
    X_r = pca.fit(x).transform(x)
    # Percentage of variance explained for each components
    print("PCA explained variance ratio (first two components): {}".format(
        str(pca.explained_variance_ratio_)))
    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(X_r[:, 0], X_r[:, 0], c=y, alpha=0.8)
    ax.set_title("PCA of wine dataset")

    kpca = KernelPCA(n_components=2, kernel='rbf')
    X_kpca = kpca.fit(x).transform(x)
    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(X_kpca[:, 0], X_kpca[:, 0], c=y, alpha=0.8)
    ax.set_title("kPCA of wine dataset")

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(x, y).transform(x)
    # Percentage of variance explained for each components
    print("LDA explained variance ratio (first two components): {}".format(
        str(lda.explained_variance_ratio_)))
    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(X_r2[:, 0], X_r2[:, 0], c=y, alpha=0.8)
    ax.set_title("LDA of wine dataset")

    plt.show()


if __name__ == '__main__':

    # read file
    filename = "winequality-red.csv"
    df = pd.read_csv(filename, delimiter=';')
    headers = list(df.columns)

    # assign columns to variables
    x = df[headers[:-1]]
    y = df[headers[-1]]

    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.2, random_state=1)
    linearRegression()
    classification()
    clustering(showPlots=True)
    dimensionalityReduction()
