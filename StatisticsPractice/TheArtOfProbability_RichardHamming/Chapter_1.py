'''
Excercise 1.9-1
Consider the Birthday Problem except that you ask for duplicate days of the month (with 30 day months).

Equation is Probability = 1 - (30/30)*(29/30)...((30-k+1)/30)
And it only works with whole numbers, so P(5), P(6), for P(k people)

Check for probability for number of people up to 30, because with 31+ obviously people
will share a birthday.
'''
def excercise_1(print_result=False):
    for person_index in range(1,31):
        not_probability = 1
        for index in range(1,person_index+1): # This range ensures the range starts from 1 person to a max of 30
            not_probability *= ((30-index+1)/30)
        if print_result:
            print("Probability of {} people is {:.4f}".format(person_index,1-not_probability))
    return

excercise_1(print_result=False)


'''
Excercise 1.9-2
This is a coincidence problem. So out of 10 or n choices, what is the probability
that none or one of the same of n is chosen?

It is the same as the birthday or month problem (above), except instead of 365 or 30 days, you have n of something.
'''
n = int(input("what is the sample size (n)? "))
for main_index in range(1,n+1):
    not_probability = 1
    for index in range(1,main_index+1): # This range ensures the range starts from 1 person to a max of 30
        not_probability *= ((n-index+1)/n)
    print("Probability of {} is {:.4f}".format(main_index,1-not_probability))