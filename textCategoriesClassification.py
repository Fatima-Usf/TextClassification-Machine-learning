import sklearn.datasets as skd


#Loading the data
categories = ['talk.politics.misc', 'rec.sport.baseball','comp.graphics', 'sci.electronics']

data_train = skd.load_files('/Users/mac/Documents/Coding/ML/Bays/20news-bydate/20news-bydate-train', categories= categories, encoding= 'ISO-8859-1')
data_test = skd.load_files('/Users/mac/Documents/Coding/ML/Bays/20news-bydate/20news-bydate-test',categories= categories, encoding= 'ISO-8859-1')

from sklearn.feature_extraction.text import CountVectorizer

""" a little example to understand the methods M using

#my text
text = ["I love cats more than anyone in the world.",
        "The cat.",
        "world"]
#give a unique id for every word and then count how much this id is frequent in each catégori
counter = CountVectorizer()
counter.fit(text)
print("Vocabulary "+str(counter.vocabulary_))
counter.get_feature_names()
print("features names: "+str(counter.get_feature_names()))

counts = counter.transform(text)
print("Counts shape :"+str(counts.shape))
print("Counts:"+str(counts.toarray())) """


counterV = CountVectorizer()
x_train = counterV.fit_transform(data_train)
x_train.shape

from sklearn.feature_extraction.text import TfidTransformer

transformer = TfidTransformer()
x_trainTfid = transformer.fit_transform(x_train)
x_trainTfid.shape

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(x_trainTfid, x_train)

#Teste
x_teste = counterV.transform(data_test.data)
x_testeTfid = transformer.transform(x_teste)
predicted = clf.predict(x_testeTfid)

from sklearn import metrics
from sklearn.metrics import accuracy_score
print("Accuray:", accuracy_score(data_test.target, predicted))
print(metrics.classification_report(data_test.target, predicted, target_names=data_test.target_names))


vectorizer.fit(counts)
print("Learnin frequency of all features:" +str(vectorizer.idf_)+'\n\n')

freq = vectorizer.transform(counts)




