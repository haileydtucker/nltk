import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
%matplotlib inline
import nltk
import string
# May need this if not already installed
nltk.download('stopwords')
nltk.download('punkt')

#Downloading csv as a panda's df
df = pd.read_csv('dmml_survey.csv')

df = df.dropna()

impressions = df.Impressions

impressions

#making a list of lists

all_docs = []
stemmer = nltk.stem.porter.PorterStemmer()

for doc in impressions:
# Make lowercase
    lowered = doc.lower()
# Remove punctuation
    fixed = lowered.translate(str.maketrans('', '', string.punctuation))
# Tokenize (splitting on whitespace, plus a few extras)
    tokens = nltk.word_tokenize(fixed)
# Remove stop words
    nonstop = []
    for w in tokens:
        if w not in nltk.corpus.stopwords.words('english'):
            nonstop.append(w)
# Stemming
    tokens = []
    for w in nonstop:
        tokens.append(stemmer.stem(w))
# Add the current list of tokens to the list of all the lists of tokens
    all_docs.append(tokens)
num_docs = len(all_docs)
print('Processed', num_docs, 'documents.')
print('First doc looks like this:', all_docs[0])

# Build a frequency distribution from the TextCollection vocab

# TextCollection

tc = nltk.TextCollection(all_docs)

fdist = tc.vocab()
unique_terms = list(fdist.keys())
num_terms = len(unique_terms)

# Make a 2D array to hold the TF-IDF scores
TD = np.zeros((num_docs, num_terms))
# Note that there are a couple ways to speed this next part up,
# but they sacrifice clarity for speed.
# Loop through each document
for i in range(num_docs):
# Loop through each term
    for j in range(num_terms):
# Grab the current document from the list of all the documents
        doc = all_docs[i]
# Grab the current term from the list of all the unique terms
        term = unique_terms[j]
# Calculate the TF-IDF score for the current term and document
# and store it in the TD array
        TD[i, j] = tc.tf_idf(term, doc)
print('TD matrix created.')
print('The columns correspond to the terms:')
print(unique_terms)
print('\nThe first document has these TD-IDF scores:')
print(TD[0, :])

# Visualize the term-document matrix
width = 60
height = 5
fig = plt.figure(figsize=(width, height))
ax = fig.add_subplot(111)
cax = ax.matshow(TD[:,:80])
fig.colorbar(cax)
plt.xticks(np.arange(80), unique_terms, rotation=90)
plt.yticks(np.arange(num_docs), np.arange(num_docs))
plt.xlabel('Term')
plt.ylabel('Document')
plt.show()

# Calc distance matrix

dist = pdist(TD, metric='cosine')
dist = squareform(dist)
# Convert cosine distance to similarity
# (so that bigger numbers mean more similar)
sim = 1 - dist
print(sim.shape)
sim

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111)
cax = ax.matshow(sim)
fig.colorbar(cax)
plt.xticks(np.arange(num_docs), np.arange(num_docs))
plt.yticks(np.arange(num_docs), np.arange(num_docs))
plt.xlabel('Document')
plt.ylabel('Document')
plt.show()
