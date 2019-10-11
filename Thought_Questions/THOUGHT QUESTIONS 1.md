# THOUGHT QUESTIONS 1

1. When the professor talked about the curse of dimensionality in section 1.5 example 8, she mostly explained the dimensional explosion situation for discrete multidimensional random variables. Another scenario for the curse of dimensionality is to describe that the classifier is harder to be trained (problem is harder to be solved) for higher dimensional/feature data. If so, why do people normally collect more features for data to avoid partial observability? Say if we have an n-dimensional dataset and simply copy and paste all the existing dimensions/features and expand the dataset into a 2n-dimensional one, will this increase the difficulty for training the classifier?
2. The expected cost consists of an irreducible error (mainly due to the data noise and partial observability) and a reducible error (the distance between the current model and optimal model). In class, the professor mentioned over-fitting is the case when the total cost is lower than the irreducible cost. I'm wondering how is that possible (the total cost is lower than the irreducible cost) considering both reducible cost term $E[(f(x)-E[Y|x]^2)]$ and irreducible cost term $E[(E[Y|x]-y)^2]$ are equal or greater than 0.
