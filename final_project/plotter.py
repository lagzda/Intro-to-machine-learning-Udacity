import numpy
import scipy
import scipy.stats
import matplotlib.pyplot as plt


def confidence_interval(x1, f1, f2):
    # generate two normally distributed 2d arrays
    # x1=numpy.random.multivariate_normal((100,420),[[120,80],[80,80]],1000)
    # fit a KDE to the data
    pdf1=scipy.stats.kde.gaussian_kde(x1.T)

    # create a grid over which we can evaluate pdf
    q,w=numpy.meshgrid(range(50,200,10), range(300,500,10))
    r1=pdf1([q.flatten(),w.flatten()])

    # sample the pdf and find the value at the 95th percentile
    s1=scipy.stats.scoreatpercentile(pdf1(pdf1.resample(1000)), 1)

    # reshape back to 2d
    r1.shape=(20,15)

    # plot the contour at the 95th percentile
    plt.contour(range(50,200,10), range(300,500,10), r1, [s1],colors='b')

    # scatter plot the two normal distributions
    plt.scatter(x1[f1],x1[f2],alpha=0.3)

    plt.show()