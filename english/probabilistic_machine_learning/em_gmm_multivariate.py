"""
Created on Mon May 10 17:17:25 2021

@author: Anuj Shah
"""

import silence_tensorflow.auto
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions

from tqdm import tqdm # progress-meter

# to create a mock dataset use sklearn make_blobs
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


tf.random.set_seed(42)
np.random.seed(42)

# prepare mock data and get their parameters used to initialize the distribution
# as in the univariate case
def mock_data(n_samples,n_classes):
    # prepare dataset with two clusters
    x,y=make_blobs(n_samples=n_samples,centers=n_classes,shuffle=False,cluster_std=0.7,random_state=2)
    #x=x[::-1]
    # plot the dataset, you can see two clusters
    plt.figure(figsize=(8,8))
    plt.scatter(x[:,0],x[:,1],marker='x')
    plt.title('The sklearn created dataset')
    plt.xlabel('X1')
    plt.ylabel('X2')


    
    #compute the parameters of the two clusters that you will need to 
    #recreate this distrubution using tfensorflow proability
    
    mean_gt=[]
    sigma_gt=[]
    class_prob_gt=[]
    for i in range(n_classes):
        x_sub_idx=np.where(y==i)
        x_sub=x[x_sub_idx]
        mu,sig=estimate_params(x_sub)
        prob=len(x_sub)/n_samples
        mean_gt.append(mu.astype('float64'))
        sigma_gt.append(sig.astype('float64'))
        class_prob_gt.append(prob)
        
    return mean_gt,sigma_gt,class_prob_gt
  
# function to compute mean and std of the mock data
def estimate_params(x_sub):
    num_sub_samples=x_sub.shape[0]
    sum_ = np.sum(x_sub,axis=0)
    mu = (sum_/num_sub_samples)
    var = np.var(x_sub,axis=0)
    sigma=np.sqrt(var)
    return mu,sigma



def em(datasets, n_classes, n_iterations, random_seed):
    
    n_samples = datasets.shape[0]

    np.random.seed(random_seed)

    # Initial guesses for the parameters
    # Although you can randomly choose the initial parameters
    # its better to start with some educated guess
    # to guess the low and high for mean
    # low = np.min(datasets)
    # high = np.max(datasets)
    # for sigma the wisest in can vary is 3-(-11)=14
    mus=np.random.randint(low=-11,high=3,size=(n_classes,datasets.shape[1])).astype('float64')
    sigmas=np.random.randint(low=0,high=14,size=(n_classes,datasets.shape[1])).astype('float64')
    class_probs = np.random.dirichlet(np.ones(n_classes))

    for em_iter in tqdm(range(n_iterations)):
        # E-Step

        responsibilities = tfp.distributions.MultivariateNormalDiag(loc=mus, 
                                scale_diag=sigmas).prob(datasets.reshape(-1,1,2)).numpy() * class_probs
        
        responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1, keepdims=True)

        class_responsibilities = np.sum(responsibilities, axis=0)

        # M-Step
        class_probs = class_responsibilities / n_samples
        for c in range(n_classes):
            mus[c] = np.sum(responsibilities[:, c].reshape(-1,1) * datasets,axis=0) / class_responsibilities[c]
            sigmas[c] = np.sqrt(
                np.sum(responsibilities[:, c].reshape(-1,1) * (datasets - mus[c])**2,axis=0) / class_responsibilities[c]
            )
    
    return class_probs, mus, sigmas



def main():

    random_seed = 42 # for reproducability
    n_samples = 500
    n_iterations = 1000
    n_classes = 2
    
    params=mock_data(n_samples,n_classes)
    
    mus_true = np.array(params[0])
    sigmas_true = np.array(params[1])
    class_probs_true = np.array(params[2])
    print('True mean: ', mus_true)
    print('True sigma: ', sigmas_true)
    print('True class prob: ', class_probs_true)


    # generate the distribution from whihch we will generate the data
    bivariate_gmm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=class_probs_true),
        components_distribution=tfp.distributions.MultivariateNormalDiag(
            loc=mus_true,
            scale_diag=sigmas_true,
        )
    )

    # REMARK: I added the random_seed after the video as I unfortunately missed
    # it
    datasets = bivariate_gmm.sample(n_samples, seed=random_seed).numpy()
    
    
    # scatter plot, to see the dataset
    plt.figure(figsize=(8,8))
    plt.scatter(datasets[:,0],datasets[:,1],marker='x')
    plt.title('created dataset using tfp')
    plt.xlabel('X1')
    plt.ylabel('X2')
    # print(dataset)
    
    # further you can do 3d plot to see two peaks
    x1,x2=np.meshgrid(datasets[:,0],datasets[:,1])
    data=np.stack((x1.flatten(),x2.flatten()),axis=1)    
    p=bivariate_gmm.prob(data).numpy()
    
    plt.figure(figsize=(8,8))
    ax=plt.axes(projection='3d')
    plt.contour(x1,x2,p.reshape(x1.shape))
    ax.plot_surface(x1, x2, p.reshape(x1.shape), cmap='viridis')
    plt.title('3d plot showing two distributions')
        
    # Now assuming that we just hav dataset and we don't know the distribution 
    # parameters lets use Expectation-Maximization for estimating the paameters

    class_probs, mus, sigmas = em(datasets, n_classes, n_iterations, random_seed)

    print('Estimated mean',mus)
    print('Estimated sigma',sigmas)
    print('Estimated_class probs',class_probs)
    print('\n')
    print('--comparing with the true values--')
    print('\n')
    print('True mean: ', mus_true)
    print('True sigma: ', sigmas_true)
    print('True class prob: ', class_probs_true)

    plt.show()

if __name__ == "__main__":
    main()
    
'''
for me the results were
Estimated mean [[-1.28429317 -9.50851172]
 [ 0.94914667 -1.4187643 ]]
Estimated sigma [[0.69109055 0.724092  ]
 [0.68222985 0.70103666]]
Estimated_class probs [0.512 0.488]

--comparing with the true values--

True mean:  [[-1.28003044 -9.50119473]
 [ 0.98079739 -1.39194926]]
True sigma:  [[0.72702424 0.71221927]
 [0.67052781 0.68750853]]
True class prob:  [0.5 0.5]
'''