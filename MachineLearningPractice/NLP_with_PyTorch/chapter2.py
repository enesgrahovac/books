'''
tokenizing text
'''
import spacy
nlp = spacy.load('en')
text = u"Mary, don't slap the green witch"
print([str(token) for token in nlp(text.lower())])

from nltk.tokenize import TweetTokenizer ## Gets the tokens from a tweet, including emojis, ats, and hashtags
tweet=u"Snow White and the Seven Degrees #MakeAMovieCold@Midnight :-)"
tokenizer = TweetTokenizer()
print(tokenizer.tokenize(tweet.lower()))

'''
N-Grams are fixed-length amount of tokens in a text.
Function below generates N-Grams of an inputted text and N value
'''
def n_grams(text,n):
    return [text[i:i+n] for i in range(len(text)-n+1)]

cleaned = ['mary', ',', "n't", 'slap', 'green', 'witch', "."]
print(n_grams(cleaned,3))

'''
Lemmas:
A lemma is a base word.
For example "flow, flew, flies, flown, flowing ... " all have the lemma "fly"

Lemmatization is reducing words to their root forms.
'''

doc = nlp(u"he was running late")
for token in doc:
    print('{} --> {}'.format(token, token.lemma_))

'''
POS - Part-of-Speech - tagging
'''
doc = nlp(text)
print("\nPOS Tagging: ")
for token in doc:
    print('{} - {}'.format(token,token.pos_))

'''
Chunking : splits a text by nouns, verbs, adjectives, etc.

Named Entity Recognition: identifies entities like person, location, organization, etc.

'''
print("\nNoun Chunking:")
doc = nlp(u"Mary slapped the green witch.")
for chunk in doc.noun_chunks:
    print('{} - {}'.format(chunk,chunk.label_))