import numpy as np
import pandas as pd
import tensorflow as tf
#rating.csv consists of index as users and columns as nof of dishes with lot of unknowns
rates = pd.read_csv('ratings.csv',index_col='user')
#argwhere collects the indices of the dataframe where the value is not nan
inds = np.argwhere(~rates.isnull().values)
data = rates.values
#seperates values and nan using indices
vals = tf.gather_nd(data,inds)
#creating sparse tensor with specific indices and values
sparse = tf.SparseTensor(inds,vals,data.shape)
#two random matrices.ie weights
a = tf.Variable(np.random.random((data.shape[0],50)))
b = tf.Variable(np.random.random((50,data.shape[1])))
c = tf.matmul(a,b)

values = tf.gather_nd(-c,inds)
c_sparse = tf.SparseTensor(inds,values,c.shape)
#calucalte error by sub original sparse - computed sparse matrix
err = tf.sparse_add(sparse,c_sparse)
#mse 
loss = tf.sparse_reduce_sum(tf.square(err))
#optimizing 
opt = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#run the following till loss gets minimum
for i in range(10000):
    _,los =sess.run([opt,loss]) 
    print(i,los)
#the predicted ratings for the unknown ratings
predicted = sess.run(c)

