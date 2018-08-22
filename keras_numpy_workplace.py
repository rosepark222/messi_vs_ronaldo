import numpy as np
x = np.sort ( np.random.rand(1, 10) )
#x_sort = np.sort(x)
print (x)
print (x.T)

slope = .5 * np.random.randn(1) + 1
print (slope )
intercept = .2 * np.random.randn(1)
print (intercept)

y = slope * x + intercept
print(y)

print ("vstack")
xy = np.vstack((x,y))
print( xy )
print ("reshaping 10,2")
print( np.reshape(xy, (10, 2)) ) 

print ("hstack of transposed x and y")
xy = np.hstack((x.T,y.T))
print( xy )

# a = np.array([[1,4],[3,1]])
# b = np.sort(a)
# print(b)

def messi_gen():
   x = np.sort ( np.random.rand(1, 10) )
   slope = .5 * np.random.randn(1) + 1
   intercept = .2 * np.random.randn(1)
   y = slope * x + intercept
   return (np.hstack((x.T,y.T)))

def ronaldo_gen():
   x = np.sort ( np.random.rand(1, 10) )
   slope = -1* (.5 * np.random.randn(1) + 1)
   intercept = .2 * np.random.randn(1) + 1
   y = slope * x + intercept
   return (np.hstack((x.T,y.T)))


print( messi_gen() )
a = ronaldo_gen() 
b = np.array([[1,   1], [1,   1],[1,   1],[1,   1],[1,   1],  [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  
c = a*b
print(a)
print(c)
