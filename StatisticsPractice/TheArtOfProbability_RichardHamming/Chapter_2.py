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
    return_factorial = 1

    if (n>0):
        for i in range(1,n+1):
            return_factorial *= i

    # return np.product([i for i in range(1,n+1)]) ## This did not work for factorial(52)

    return return_factorial


def P(n,k): # Permutation function
    return factorial(n)/factorial(n-k)

def C_multi(n,n_k=[]): # Multinomial coefficients, n_k are the repeated values
    return int(P(n,n)/np.product([factorial(i) for i in n_k]))

# print(P(10,3))

# print(P(10,10))

# print(C_multi(7,[3,2]))
# print(C_multi(11,[4,4,2]))

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

'''
A combination is a permutation when the order is ignored.

A special case of the multinomial coefficients occurs when there are only two kinds of items to be selected;
this is the binomial coefficients C(n,k)

The formula (where n is the total items, and k items are selected):
C(n,k) = factorial(n)/(factorial(k)*factorial(n-k)) = P(n,k)/factorial(k) = C(n,n-k)

C(n,0) = C(n,n) = 1
'''

def C(n,k):
    return P(n,k)/factorial(k)

# print(C(7,3))

'''
What is the probability of a bridge hand having no cards other than 2,3,4,5,6,7,8,9,10?

Rules to bridge: https://www.acbl.org/learn_page/how-to-play-bridge/
In bridge a player gets 13 cards to start.

probability will be:
C(36,13)/C(52,13) which is the number of those numbered cards divided by all possible card combos.
'''

print(C(36,13)/C(52,13))
