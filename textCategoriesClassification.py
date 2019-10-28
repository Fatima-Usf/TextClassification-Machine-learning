import sklearn.datasets as skd


#Loading the data
categories = ['talk.politics.misc', 'rec.sport.baseball','comp.graphics', 'sci.electronics']

data_train = skd.load_files('/Users/mac/Documents/Coding/ML/Bays/20news-bydate/20news-bydate-train', categories= categories, encoding= 'ISO-8859-1')
data_test = skd.load_files('/Users/mac/Documents/Coding/ML/Bays/20news-bydate/20news-bydate-test',categories= categories, encoding= 'ISO-8859-1')


from sklearn.feature_extraction.text import CountVectorizer

text = ["I love cats more than anyone in the world.",
        "The cat.",
        "world"]
counter = CountVectorizer()
counter.fit(text)
print("Vocabulary "+str(counter.vocabulary_))
counter.get_feature_names()
print("features names: "+str(counter.get_feature_names()))

counts = counter.transform(text)
print("Counts shape :"+str(counts.shape))
print("Counts:"+str(counts.toarray()))
