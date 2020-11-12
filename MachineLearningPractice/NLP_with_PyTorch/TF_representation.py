import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

corpus = ['Time flies flies like an arrow.',
            'Fruit flies like a banana.']

'''
TF Representation Example 1-1


'''

def example_1_1(show=False):
    global vocab
    one_hot_vectorizer = CountVectorizer(binary=True)
    one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
    vocab = one_hot_vectorizer.get_feature_names()
    if show:
        heatmap = sns.heatmap(one_hot,annot=True,cbar=False,xticklabels=vocab,yticklabels=['Sentence 1', 'Sentence 2'])
        plt.show()
    return

example_1_1()

'''
TF Representation Example 1-2
'''

from sklearn.feature_extraction.text import TfidfVectorizer

def example_1_2(show=False):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
    sns.heatmap(tfidf,annot=True, cbar=False, xticklabels=vocab,yticklabels=['Sentence 1', 'Sentence 2'])
    if show:
        plt.show()
    return

example_1_2(show=True)




