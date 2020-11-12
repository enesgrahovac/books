'''
Permutations: Different ways to select k different things from a selection of n.
P(n,k) is the common notation.

P(n,k) = (n)!/(n-k)!

P(n,n) = n!

Multinomial Coefficients works when there are identical terms that can be selected that would
cause uniform permutations.

Example: arranging the letters of "success" in all possible orders, yet the s's and c's repeat.
Solutions: P(7,7)/(3!2!1!1!), where the denominator should add up to 7 for all 7 letters: 3+2+1+1 = 7.
'''

import numpy as np

def factorial(n):
    return int(np.product([i for i in range(1,n+1)]))

def P(n,k): # Permutation function
    return factorial(n)//factorial(n-k)

def C_multi(n,n_k=[]): # Multinomial coefficients, n_k are the repeated values
    return int(P(n,n)/np.product([factorial(i) for i in n_k]))

print(P(10,3))

print(P(10,10))

print(C_multi(7,[3,2]))
print(C_multi(11,[4,4,2]))

'''
How many 3 letter combinations can be made from the letters of "Mississippi"?
'''

total_combinations = 0

# Case 1: 3 distinct Letters
total_combinations += P(3,3)

# Case 2: 3 Repeated Letters
total_combinations += 2 #For i and p

# Case 3: 1 repeated letter and 1 distinct letter
total_combinations += P(3,3) * (factorial(5)/factorial(4))

print(total_combinations)
