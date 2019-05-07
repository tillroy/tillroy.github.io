---
title: "K-means clustering (Part 2)"
tags: [kmeans, k-means]
---

# Distance function

In previous [Part]({% post_url 2019-05-06-k-means-clustering-part-1 %}) we were dealing with implementation of K-means clustering algorithm ad defined part that allows us to get actual values of kernels in each cluster. Here we will be working with finishing class we defined before, so we will be able to assign cluster label to new previously unknown data value.

Let quietly remind how does our class look like:

```python
class KmeansClustering(object):
    def __init__(self, data:np.ndarray, k:int=1)->None:
        pass

    def calc_distance(self, cluster:np.ndarray, kernel:np.array)->np.array:
        # calculate distance between dataset values
        pass

    def train(self)->None:
        # calculate kernels values
        pass

    def fit(self, sample->np.array)->int:
        # compare new value and assign it to the kernel class
        pass
```

As we can see from class structure above we have 3 class methods and one magic method. We have implemented bigger part of them, only `fit` method does not defined yet. Let fix this small gap:

```python
def fit(self, sample:np.array)->int:
    assert self.kernels[0].shape == sample.shape, "Sample shape is different from dataset values"
    
    # use kernels as data input and sample as kernel
    # we are calculating distance from each kernel to sample
    res = self.calc_distance(self.kernels, sample)
    cluster_value = np.argmin(res)

    return cluster_value
```

As you could note the end of this method looks similarly to the train method(part with searching cluster labels). The idea on `fit` is to define what cluster kernel more cluster to the sample input. Basically everything we need is to calculate distance between our kernels and data sample.

> Important: Here we treat kernels as simple data and data sample as kernel, it allows calculate distance between them

Also we could enhance our code and add shape shaking part. So we could be sure that sample has the same shape as data values at dataset.

Now we could test our results:

```python
np.random.seed(1)
kmc = KmeansClustering(car_weights, k=6)
kmc.train()
cluster_label = kmc.fit(np.array([3050]))

print(kmc.kernels)
->[[2471.55555556]
  [4606.90909091]
  [3628.07142857]
  [4048.77777778]
  [3181.73333333]
  [6248.        ]]

print(cluster_label)
->4

```

As we can wee we get cluster labels #4(in Zero-based indexing) and it is really close to its kernel value ***3181.73333333***

## Data shape and distance function

We mentioned before that it will be great if we could use the same K-mean definition for different problems(e.g. YOLO anchor box clustering problem). We build our class in way that only one method is sensitive to the data input as well as distance function definition -- it is `get_distance` method. That is right now if we want to use this class with different distance function or with other data set everything we need it is overwrite distance function. Let generalize our distance function so we could use it with class inheritance:

```python
def calc_distance(self, cluster:np.ndarray, kernel:np.array)->np.array:
    """Main distance function.
    :param cluster: cluster data, column vector or matrix
    :param kernel: sample data usual one feature from dataset

    return column vector
    """
    raise NotImplementedError
```

Now our distance function becomes abstract method. We could use this feature and define case specific K-means class for car weights K-means calculation

```python
class CarKmeans(KmeansClustering):
    def calc_distance(self, cluster:np.ndarray, kernel:np.array)->np.array:
        dist = (cluster - kernel)**2
        return dist.flatten()
```

And thats is now everything we need to know it is data shape and waht exectly we want to treat as distance.

Let define one more example for popular Iris dataset:

```python
class IrisKmean(KmeansClustering):
    def calc_distance(self, cluster:np.ndarray, kernel:np.array)->np.array:
        res = []
        for el in cluster:
            dist = np.sqrt(((el[0]-kernel[0])**2) + ((el[1]-kernel[1])**2))
            res.append(dist)

        return np.array(res)
```

We defined our distance function as Euclidean distance. Let try it out

```python
from sklearn.datasets import load_iris

# load dataset
iris = load_iris()
# use only 2 features from 4 available
X = iris.data[:,1:3]

np.random.seed(1)

kmc = IrisKmean(X, k=4)
kmc.train()

print(kmc.kernels)
->[[3.86666667 1.50666667]
  [2.75087719 4.32807018]
  [3.03255814 5.67209302]
  [3.24       1.44285714]]

cluster_label = kmc.fit(np.array([4, 4]))
print(cluster_label)
->1
```

Now we could use our K-mean class for any type of problems including YOLO anchor box clustering, but it will be the topic of the next K-means part.