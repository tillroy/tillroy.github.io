---
title: "K-means clustering (Part 1)"
tags: [kmeans, k-means]
---

# K-means clustering

Sooner or later you will find K-mean clustering algorithm during traveling Data Science, Machine Learning of Artificial intelligent fields. This is one of the simples unsupervised learning algorithms. You could wondering how does K-Means algorithm work. Let check it backed and create it from scratch.

You could find a lot of examples all around the Internet, because of that let build this algorithm in a bit different way, when it could helps us during implementation of YOLO algorithm, for example. At this time let define it as separate class rather than just a function. Here is abstract structure we are gonna to use:

```python
class KmeansClustering(object):
    def __init__(self):
        pass

    def train(self):
        # calculate kernels values
        pass

    def fit(self, sample):
        # compare new value and assign it to the kernel class
        pass
```

The main idea behind class definition that we can save trained results and use them later.

## K-means background

This algorithm allows us assign split all data into defined amount of groups or clusters. I supposes that `K` in naming comes from `kernel` which is main conceptual idea behind this algorithm. Basically each cluster has it's own kernel(value) which represent all cluster.

In my opinion we could compare meaning of `kernel` with `average` or `mean` in statistic. When we have a set of data values, e.g. car weight, and would like to represent all set by one value, the easies thing we could do it is calculate the mean of all set:

Model | Weight
--- | ---
2012 Toyota Camry | 3,190 pounds
2012 Toyota Prius | 3,042 pounds
2012 Toyota Avalon | 3,572 pounds
2013 Toyota Matrix | 2,888 pounds
2013 Chevrolet Equinox LS | 3,777 pounds
2013 Chevrolet Corvette | 3,208 pounds
2013 Chevrolet Malibu | 3,393 pounds
2012 Chrysler Town and Country | 4,652 pounds
2012 Subaru Outback | 3,495 pounds
... | ...


> Note: data taken from [here](https://cars.lovetoknow.com/List_of_Car_Weights)

```python
car_weights = np.array([
[3190], [3042], [3572], [2888], [3777], [3208], [3393], [4652], [3495], [3208], [4344], [2617], [5949], [2535], [4756], [2396], [2701], [3084], [3102], [3600], [3756], [3300], [3496], [3668], [3780], [3682], [3814], [2768], [3540], [2354], [2935], [4037], [1808], [3323], [3968], [3540], [3295], [4233], [3532], [2512], [4646], [4742], [4047], [3256], [6547], [4553], [3950], [4004], [4029], [3393], [3541], [4979], [4740], [3941], [4398], [4470], [2553], [3109], [4396], [4230]
])

cars_mean = car_weights.mean()
print(cars_mean)
-> 3672.9
```

This one number does not make much sense, but we could be sure for example that this weight value will be bigger that average humans weight and lover than average tracks weight. So this value gives us some generalized value for all category(cluster) -- cars. Exactly this value we should treat as cluster `kernel`.

> Note: let use column vector instead of simple list, it allows us use more features in feature not just one car weight(e.g length, height)

In this example we have only one cluster, so it was easy. We just calculate mean of all data values. What if we want to define some specific subcategories for car dataset we already have. Of course we could explore the data or calculate some statistics, but we also could use the K-means. Let use the next car classes from the same [post](https://cars.lovetoknow.com/List_of_Car_Weights):

Car Class | Weight
Compact car | 2,979 pounds
Midsize car | 3,497 pounds
Large car | 4,366 pounds
Compact truck or SUV | 3,470 pounds
Midsize truck or SUV | 4,259 pounds
Large truck or SUV | 5,411 pounds

This data has been already calculated. But what if we don't have this weights values, and we know only number of classes we want to split our dataset to. For example, from table above we want to define ***6*** car classes based on our data. K-means could calculate for us 6 means which represent each car class in all dataset.

> Important: K-mean requires the number of clusters the dataset should be split to as input!

### What does K-means do?

1. take amount of clusters we want as input
2. initialize clusters kernels, also known as cluster centroids(usialy randomly defined or taken from dataset)
3. calculating distance from each kernel to each data value(distance function could be different)
4. split data into clusters based on distance metric(assign data to the cluster with smallest distance metric)
5. calculate some generalized value for each cluster(in most cases it is mean value of the cluster)
6. repeat everything until some action occur(it could be number of iterations, some threshold in kernel changing from one iteration to another)

### How to build K-means from scratch?

Let's take the class template from the very beginning of the post and add there a bit of logic:

```python
class KmeansClustering(object):
    def __init__(self, data:np.array, k:int=1)->None:
        """Initialize class attributes.

        :param data: array of data values
        :param k: number of kernels/clusters we want to calculate
        """
        self.data = data
        # 1. take amount of clusters we want as input
        self.k = k

    def train(self):
        # total number of samples in dataset
        samples_amount = self.data.shape[0]

        # randomly get k data sample indices
        init_k_inices = np.random.choice(samples_amount, self.k, replace=False)

        # 2. initialize clusters kernels
        # take values from dataset based on generated indices
        self.kernels = self.data[init_k_inices]
```

Let check the values:

```python
# we use seed value to simulate the same random output
np.random.seed(1)

kmc = KmeansClustering(car_weights)
kmc.train()
print(kmc.kernels)
-> [[2512]]

# for 2 kernels
kmc = KmeansClustering(car_weights, k=2)
kmc.train()
print(kmc.kernels)
->[[2512] [4742]]
```

Next we should calculate distance between our kernel and all data values. We could use different methods, but for this tutorial let make it from scratch too. We will use ***squared difference*** at this moment for preventing negative values.

```python
def get_distance(self, cluster, kernel):
    dist = (cluster - kernel)**2
    return dist.flatten()
```

At this moment let focus at `train` method

```python
def train(self):
        # total number of samples in dataset
        samples_amount = self.data.shape[0]

        # initialize distances container with shape (sample_amount, k)
        # this matrix contains distances from each kernel to each data value
        # kernel index corresponds to matrix column
        self.distances = np.empty((samples_amount, self.k))

        # 2. initialize clusters kernels
        self.kernels = self.data[np.random.choice(samples_amount, self.k, replace=False)]
        # because we have inly integers in our dataset we have to change data type to see correct results
        self.kernels = self.kernels.astype(float)

        for k_ind in range(self.k):
            kernel = self.kernels[k_ind]
            # 3. calculating distance from each kernel to each data value
            kernel_distances = self.get_distance(self.data, kernel)
            # write kernel distances into respective column
            self.distances[:,k_ind] = kernel_distances

            #get clusters
            # return cluster index with smallest distance value across all clusters for the same sample
            self.clusters = np.argmin(self.distances, axis=1)

        for k_ind in range(self.k):
            # 4. split data into clusters based on distance metric
            cluster_data = self.data[self.clusters == k_ind]
            # 5. calculate some generalized value for each cluster
            self.kernels[k_ind] = np.mean(cluster_data, axis=0)
```

We will update this method if feature, but not let check intermediate results:

```python
np.random.seed(1)

kmc = KmeansClustering(car_weights, k=1)
kmc.train()

print(kmc.kernels) 
->[[3672.9]]

# 2 kernels
->[[3071.4375    ]
   [4360.28571429]]

# 3 kernels
->[[2592.41666667]
   [4775.66666667]
   [3564.54545455]]
```

As we can see for one kernel, classifier returns exactly the same mean value among all cluster values, cluster in this case is equal to the dataset itself. Also it works for 2,3 and more kernels, but it is only one iteration, let fix it and put everything in a loop:

```python
def train(self):
    samples_amount = self.data.shape[0]

    self.distances = np.empty((samples_amount, self.k))

    self.kernels = self.data[np.random.choice(samples_amount, self.k, replace=False)]
    self.kernels = self.kernels.astype(float)

    for el in range(100):
        for k_ind in range(self.k):
            kernel = self.kernels[k_ind]
            kernel_distances = self.get_distance(self.data, kernel)
            self.distances[:,k_ind] = kernel_distances

            self.clusters = np.argmin(self.distances, axis=1)

        for k_ind in range(self.k):
            cluster_data = self.data[self.clusters == k_ind]
            self.kernels[k_ind] = np.mean(cluster_data, axis=0)
```

```python
np.random.seed(1)

kmc = KmeansClustering(car_weights, k=3)
kmc.train()

print(kmc.kernels)
->[[2592.41666667]
   [4775.66666667]
   [3564.54545455]]
```

We run this process 100 times but get the same results as ruining it only 3 times. Thats is right after certain amount of iterations kernels are not changed any more, it is common practice to check kernel values during the iterations and if they are not changing stop the calculation process. But first of all let check what we will get for ***6*** classes.

```python
->[[2471.55555556]
   [3181.73333333]
   [3628.07142857]
   [4048.77777778]
   [4606.90909091]
   [6248.        ]]
```

Let put it into table and check how far we are from original data:

Model | Original weights | Calculated weights
--- | --- | ---
Compact car | 2,979 pounds | 2471.55555556
Compact truck or SUV | 3,470 pounds | 3181.73333333
Midsize car | 3,497 pounds | 3628.07142857
Midsize truck or SUV | 4,259 pounds | 4048.77777778
Large car | 4,366 pounds | 4606.90909091
Large truck or SUV | 5,411 pounds | 6248.0

As we can see we are really close to original data, of course the is difference but we get similar results only from ***60*** samples and one feature. For sure K-mean does not pretend to be car classifier, but this example should show how easy we could split dataset into defined number of clusters based on one or more features.

Now we could fix iteration problem

```python
class KmeansClustering(object):
    def __init__(self, data:np.array, k:int=1)->None:
        """Initialize class attributes.

        :param data: array of data values
        :param k: number of kernels/clusters we want to calculate
        """
        self.data = data
        self.k = k
        self.clusters = None
        self.distances = None
        self.kernels = None

    def get_distance(self, cluster, kernel)->np.array:
        dist = (cluster - kernel)**2
        return dist.flatten()

    def train(self):
        samples_amount = self.data.shape[0]
        self.distances = np.empty((samples_amount, self.k))
        self.kernels = self.data[np.random.choice(samples_amount, self.k, replace=False)]
        self.kernels = self.kernels.astype(float)

        # init placeholder for clusters data
        prev_clusters = np.zeros((samples_amount,))
        while True:
            for k_ind in range(self.k):
                kernel = self.kernels[k_ind]
                kernel_distances = self.get_distance(self.data, kernel)
                self.distances[:,k_ind] = kernel_distances
                self.clusters = np.argmin(self.distances, axis=1)

            for k_ind in range(self.k):
                cluster_data = self.data[self.clusters == k_ind]
                self.kernels[k_ind] = np.mean(cluster_data, axis=0)

            if (prev_clusters == self.clusters).all():
                break

            prev_clusters = self.clusters
```

We define K-means classifier in this section and check how it works, let continue in [Part 2]({% post_url 2019-05-07-k-means-clustering-part-2 %}).