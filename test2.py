import numpy as np

def membership(x, a, b, c, d):
    return max(0, min((x - a) / (b - a), 1 - (x - c) / (d - c)))

# Example usage
a = 10
b = 20 
c = 25
d = 35

temperatures = np.arange(0, 40)
memberships = [membership(x, a, b, c, d) for x in temperatures]

print(temperatures)
print(memberships)

import numpy as np

# Prior distribution 
def prior(theta):
    return 1.0  

# Likelihood function   
def likelihood(data, theta):
    return 1.0


# Posterior distribution via Bayes' theorem
def posterior(theta, data):
    return likelihood(data, theta) * prior(theta)

# Normalize the posterior
def normalize(unnormalized):
    return unnormalized / np.sum(unnormalized) 

# Compute the posterior mean / expectation
def posterior_mean(data):
    thetas = np.linspace(-10, 10, 100)
    posterior_probs = normalize(posterior(thetas, data))
    return np.sum(thetas * posterior_probs)

# Example usage  
import numpy as np 

# Same functions defined above 

# Example usage
observed_data = [1.2, 2.1, -0.5]  

data = np.array(observed_data)

print("Posterior mean:", posterior_mean(data))

data = [1.05, 0.98, 1.02, 0.95, 1.01] # Example measured data

import numpy as np

def f(x, a, b, c):
    """General membership function placeholder"""
    return 0

def membership(x, a, b, c):
    """Computes the degree of membership of x"""   
    return f(x, a, b, c)


# Example triangular membership function
def tri_func(x, a, b, c):
    """Triangular membership function"""
    
    if x <= a:
        return 0
    elif a < x < b:
        return (x - a) / (b - a)
    elif x == b: 
        return 1
    elif b < x < c: 
        return (c - x) / (c - b)
    else:
        return 0
        
# Example usage        
x_val = 5 
tri_params = [2, 4, 6]  

x_val = 5  
tri_params = [2, 4, 6]   

print(membership(x_val, *tri_params)) # prints 0.5

import numpy as np

def membership(x):
    """Membership function"""
    return 0.8 * x[0]

def mu_V(x):
    """Membership function"""
    return 0.8 * x[0]

def h_V(z):
    """Support function"""
    return max(0, z)

def support(x, u):
    """Returns support function value"""
    return h_V(np.dot(x, u))

def fuzzy_vector(x, u):
    """Constructs fuzzy vector for x"""
    return (membership(x), support(x, u))

# Example 
x = np.array([0.5, 1.0]) 
u = np.array([1, 0])

print(fuzzy_vector(x, u))

import numpy as np

# Fuzzy vector 
def mu_V(x):
    """Membership function for V"""  
    return f(x, a1, a2, a3) 

def h_V(u):
    """Support function for V"""
    return np.max(np.dot(X, u))
    
