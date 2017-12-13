from sklearn.datasets import load_svmlight_file as lsf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep

# import data from the triazines.txt downloaded
# x of 186 * 60 and y of 186 * 1
Amn, bm = lsf("triazines.txt")

'''
1 Data
'''
# copy the data to np.array
A = Amn.toarray()
b = bm.copy()

# standardize the data to mean zero and deviation 1 for each column
A -= np.mean(A, axis=0, dtype=np.float64)
    # some columns has std ~ 0 where we specify them, whether do the
    # following:
#cols_sparse = np.std(A, axis=0)==0.
#A[:, ~cols_sparse] /= np.std(A[:, ~cols_sparse], axis=0)
    # or, use the sklearn method fit_transform
A = prep.StandardScaler().fit_transform(A)

# standardize the observes data to mean zero and deviation 1
b = b - np.mean(b)
b = b / np.std(b)

# sparse , can we remove the nan columns of A ?
# A = A[:, ~np.any(np.isnan(A), axis= 0)]

'''
2 Proximal operator of the indicator of the L1 ball
'''
# 2.8
# given a vector w, do the projection onto the L1 ball

def projection(w, r):
    '''
    :param w: w is the vector of the same dimension of the space
    :param r: r is the constraint (L1 radius)
    :return: return the point of P(w)
    '''
    dim = len(w)
    if dim == 0:
        return []

    # get the signs of w
    sign = np.sign(w)
    # get the obsolute value of w
    w = w * sign

    # from smallest to largest
    sorted_w = np.sort(w)
    # get the sum
    su = sum(sorted_w)

    # condition on r , solution theta exists
    if su <= r :
        return sign * w, 0

    # initiate theta
    theta = 0
    for i in range(dim):
        if su - sorted_w[i]*(dim - i) < r and r <= su:
            theta = (su - r) / (dim - i)
            #print("solve the projection problem: ")
            #print("sum", su)
            #print("theta", i, theta)
            return sign * np.maximum(0, np.array(w) - theta), theta
        else :
            su -= sorted_w[i]

# test
w = b
r = np.linalg.norm(b, ord=1) / 2
#theta is returned as the second index
print("the projection of w on ball r= ", r, "is", projection(w, r) )


'''
3 resolution of the lasso
'''
def linear_lip(A):
    return np.linalg.norm(np.dot(A.T, A), ord=2) + 1

# which is about 2000
L = linear_lip(A)
# step : about 0.0007
mystep = 2/L

def lasso_l1(A, b, step=mystep, r=1, N=1000):
    '''
    :param A:
    :param b:
    :param r:
    :return: return the solution for the lasso(A,b,r)
    '''

    n, m = A.shape
    # initiation
    x = np.zeros(m)
    # gradient descent according to f
    hist_cost = []
    hist_x = []
    hist_theta = []

    for i in range(N):
        x_old = x
        t = x - step * np.dot(A.T, np.dot(A,x) - b)
        x, theta = projection(t, r)

        # calculate the cost to set a stop
        cost = 0.5* np.square(np.linalg.norm(A.dot(x) - b, ord=2)) * 1/n
        hist_cost.append(cost)
        hist_x.append(x)
        hist_theta.append(theta)

        print("step", i, ": cost is", cost, "theta is ", theta)

        if np.linalg.norm(x - x_old, ord=2) < 1e-8:
            break
    #print("final x = ", x)

    return x, hist_cost, hist_theta, hist_x


# test
x, hist_cost, hist_theta, hist_x = lasso_l1(A, b)
plt.plot(hist_cost)

x, hist_cost, hist_theta, hist_x = lasso_l1(A, b, r=2, N=1000)
plt.plot(hist_cost)


'''4 comparison of two lasso formulations'''

'''4.2 compare two methods in one plot'''
myalpha = np.max(np.abs(A.T.dot(b))) / 2

def prox_l1(alpha, y):
    l = np.zeros(len(y))
    for i in range(len(y)):
        if y[i] < -alpha :
            l[i] = alpha + y[i]
        elif y[i] > alpha :
            l[i] = -alpha + y[i]
        else:
            l[i] = 0
    return l

def cost_f(x, A, b):
    n, m = A.shape
    return 0.5 * np.square(np.linalg.norm(A.dot(x) - b, ord=2)) * 1/n

def lasso_penal(A, b, alpha=myalpha, step=mystep, N=1000):
    n, m = A.shape
    # initiation
    x = np.zeros(m)
    # gradient descent according to f
    hist_cost = []
    hist_x = []

    for i in range(N):
        x_old = x
        t = x - step * np.dot(A.T, np.dot(A, x) - b)
        #print(t)
        x = prox_l1(alpha*step, t)
        #print(x)
        # calculate the cost to set a stop
        cost = 0.5 * np.square(np.linalg.norm(A.dot(x) - b, ord=2)) * 1/n
        hist_cost.append(cost)
        hist_x.append(x)

        print("step", i, ": cost is", cost)

        #if np.linalg.norm(x - x_old, ord=2) < 1e-8:
        if cost < 0.02:
            break
    # print("final x = ", x)

    return x, hist_cost, hist_x


x_p, hist_cost_p, hist_x_p = lasso_penal(A, b, N=1000, alpha=myalpha)
plt.plot(hist_cost_p)
plt.plot(hist_cost)

'''4.3 cross validation '''
def split_data(A, b, l):
    n, m = A.shape
    idx = np.array(range(n))
    np.random.shuffle(idx)

    test_idx = idx[0:n*l[0]]
    valid_idx = idx[n*l[0]:n*l[1]]

    A_test, b_test = A[test_idx], b[test_idx]
    A_valid, b_valid = A[valid_idx], b[valid_idx]

    return [A_test, A_valid], [b_test, b_valid]


np.random.seed(1)
As, bs = split_data(A, b, [0.7, 1])
b_t, b_v = bs[0], bs[1]
A_t, A_v = As[0], As[1]

import time

starttime = time.time()
x_p_t, hist_cost_p_t, hist_x_p_t = lasso_penal(A_t, b_t, N=1000, alpha=myalpha)
delta_time_p = time.time() - starttime

starttime = time.time()
x_l, hist_cost_l, hist_theta_l, hist_x_l = lasso_l1(A_t, b_t, r=1, N=1000)
delta_time_l = time.time() - starttime

print("1/2 mean square error for the penal term lasso is ", cost_f(x_p_t, A_v, b_v) )
print("time used for the penal term lasso:", delta_time_p )

print("1/2 mean square error for the L1 projection lasso is ", cost_f(x_l, A_v, b_v) )
print("time used for the L1 projection lasso:", delta_time_l)

''' we list the ouput here:
1/2 mean square error for the penal term lasso is  0.530712430333
time used for the penal term lasso: 0.21063685417175293
1/2 mean square error for the L1 projection lasso is  0.454060789646
time used for the L1 projection lasso: 0.21392297744750977

it seems that L1 projection is a little quicker
'''

plt.plot(hist_cost_p_t)
plt.plot(hist_cost_l)


'''1 the relation with r '''

r_lis = list(range(1,10))
def plot_by_r(r_lis):
    x_lis = []
    cos_lis = []
    hist_cost_lis = []
    for i in r_lis :
        x_i, hist_cost_i, hist_theta_i, hist_x_i = lasso_l1(A_t, b_t, r=i, N=1000)
        x_lis.append(x_i)
        cos_lis.append(cost_f(x_i, A_v, b_v))
        hist_cost_lis.append(hist_cost_i)

        plt.plot(hist_cost_i, label="r = %s" % (i))
        plt.legend()
        plt.title("values of 1/2 mean square for L1 projection formule", fontsize=16)
    return cos_lis, x_lis, hist_cost_lis
## we saw from the graph that when r is larger, the mean square values of the test set are smaller
cos_lis, _, _ = plot_by_r(r_lis)
# r-cost graph
plt.plot(r_lis, cos_lis)

## we saw from the graph that when r is larger (>=3), the validation set error becomes larger again,
## which we conclude as a phenomenon of overfitting.
### this may be more clear if we take r from 0.2 to 3
r_lis = np.arange(0.2, 3, 0.2)
cos_lis_r, _, _ = plot_by_r(r_lis)
'''
# when r is larger, the test error is smaller
# we plot the r - mean error graph to see when the r is the best
'''
# r-cost graph
plt.plot(r_lis, cos_lis_r)

'''
# and we saw clearly that r is about 1.3 minimize the cost of validation set
# and with r > 2, it<s clear overfitting
'''

'''2 the relation with alpha'''

def plot_by_alpha(alpha_lis):
    x_lis = []
    cos_lis = []
    hist_cost_lis = []
    for i in alpha_lis :
        x_i, hist_cost_i, hist_x_i = lasso_penal(A_t, b_t, N=1000, alpha=i)
        x_lis.append(x_i)
        cos_lis.append(cost_f(x_i, A_v, b_v))
        hist_cost_lis.append(hist_cost_i)

        plt.plot(hist_cost_i, label="alpha = %s" % (i))
        plt.legend()
        plt.title("values of 1/2 mean square for penal term lasso formule", fontsize=16)
    return cos_lis, x_lis, hist_cost_lis

'''run test'''
alpha_lis = np.arange(1,10,0.5)
cos_lis_a, _, _ = plot_by_alpha(alpha_lis)
'''
# when alpha is smaller, the mean square error gets smaller
# we then plot the alpha - mean error graph to get the best alpha
'''

'''plot alpha- validation cost'''
# alpha-cost graph
plt.plot(alpha_lis, cos_lis_a)

'''
# we saw that the value of alpha should be between 4 and 5, when alpha smaller than 4, it's clearly overfitting 
# and when alpha greater than 5, it's clearly underfitting. 

# Two methods get about the same smallest value for mean square , 
'''

'''5 extensions'''

'''other methods can be for example the Frank wolfe algorithm, coordinate descent algorithms.'''

