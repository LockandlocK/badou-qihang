import numpy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# Create the Document Term Matrix

def txt_to_pandas(documents):
    # Create the Document Term Matrix
    count_vectorizer = TfidfVectorizer(stop_words='english')
    #count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(documents)

    # OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix,
                      columns=count_vectorizer.get_feature_names(),
                      index=['A', 'B','C'])
    print(df)
    return df
""""
 = "love data mining"
 = "hate data mining"
"""
A = "Mr. Trump became president after winning the political election. Though he lost the support of some republican friends, Trump is friends with President Putin"

B = "President Trump says Putin had no political interference is the election outcome. He says it was a witchhunt by political parties. He claimed President Putin is a friend who had nothing to do with the election"

C = "Post elections, Vladimir Putin became President of Russia. President Putin had served as the Prime Minister earlier in his political career"

documents = [A,B,C]
wordsA = A.lower().split()
wordsB = B.lower().split()
wordsC = C.lower().split()

vocab = set(wordsA)
vocab = vocab.union(set(wordsB))
print (vocab)

vocbc = set(wordsB)
vocbc = vocbc.union(set(wordsC))
print (vocbc)

vocac = set(wordsA)
vocac = vocac.union(set(wordsC))
print (vocac)

vocab = list(vocab)
vocbc = list(vocbc)
vocac = list(vocac)



vA = numpy.zeros(len(vocab), dtype=float)
vB = numpy.zeros(len(vocab), dtype=float)
vC = numpy.zeros(len(vocab), dtype=float)

#  go through the list of features for the first sentence and increment the corresponding feature value in the vector.

for w in wordsA:
    i = vocab.index(w)
    vA[i] += 1
print(vA)

for w in wordsB:
    i = vocab.index(w)
    vB[i] += 1

print(vB)


df = txt_to_pandas(documents)
cos1 = numpy.dot(vA, vB) / (numpy.sqrt(numpy.dot(vA,vA)) * numpy.sqrt(numpy.dot(vB,vB)))

cos2 = cosine_similarity(df,df)
print(cos1,'cos1')
print(cos2, 'cos2')