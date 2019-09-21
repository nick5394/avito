from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import pandas as pd
from scipy.sparse import *
from scipy import *
import numpy as np
import time
import nltk
import re
import math
from keras.models import Input,Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import scipy
from keras.layers import Dropout
from keras.optimizers import SGD
from keras import regularizers
from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

train = pd.read_csv('C:\\Users\\Nick\\Desktop\\avito_competition\\train.csv', nrows = 1e4)
test = pd.read_csv('C:\\Users\\Nick\\Desktop\\avito_competition\\test1.csv', nrows = 1e4)

item_id = test["item_id"]

train = train.fillna(-1)
test = test.fillna(-1)

Y = np.array(train['deal_probability'])

del train['user_id'], train['item_id'], train['image'], train['image_top_1'],  train['deal_probability']

start_time = time.time()

def one_hot(column):

	list_of_unique_vals = pd.unique(column)

	column = column.apply(lambda x: list_of_unique_vals.tolist().index(x))

	rows = column.index.values
	columns = column

	data_array=np.empty(len(column)); 
	data_array.fill(1)

	matrix_shape = (rows.shape[0],len(list_of_unique_vals ))

	data = csr_matrix((data_array,(rows,columns)), shape = matrix_shape)

	return data

train_cols_to_encode = ["parent_category_name","city", "category_name", "activation_date", "user_type", "param_1", "param_2", "param_3"]
#X_use = train

#print(pd.concat([train["region"],test["region"]], axis = 0))
concat = one_hot(pd.concat([train["region"],test["region"]], axis = 0, ignore_index=True))

X = concat[:train.shape[0],:]

X_test = concat[train.shape[0]:,:]
#print(np.array(X))
#print(list(X_test))

for i in train_cols_to_encode:
	concat = pd.concat([train[i],test[i]], axis = 0, ignore_index=True)

	one_hot_col = one_hot(concat)

	X = scipy.sparse.hstack((X,one_hot_col[:train.shape[0],:]))
	X_test = scipy.sparse.hstack((X_test,one_hot_col[train.shape[0]:,:]))

#print(list(X.toarray()))
#print(list(X_test.toarray()))

max_item_seq_number = max(train["item_seq_number"])
max_price = max(train["price"])

max_item_seq_number_test = max(test["item_seq_number"])
max_price_test = max(test["price"])

train["item_seq_number"] = train["item_seq_number"].apply(lambda x: x/max_item_seq_number)
train["price"] = train["price"].apply(lambda x: x/max_price)

test["item_seq_number"] = test["item_seq_number"].apply(lambda x: x/max_item_seq_number_test)
test["price"] = test["price"].apply(lambda x: x/max_price_test)

X = scipy.sparse.hstack((X,np.array(train["price"])[:,None]))
X = scipy.sparse.hstack((X,np.array(train["item_seq_number"])[:,None]))

X_test =scipy.sparse.hstack((X_test,np.array(test["price"])[:,None]))
X_test =scipy.sparse.hstack((X_test,np.array(test["item_seq_number"])[:,None]))


#train["title"] = train["title"].apply(str.lower)

#train["title"] = train["title"].apply(lambda x: nltk.word_tokenize(x))

vocabulary=[]

#train["title"].apply(lambda x: vocabulary.extend(x))

#feed in a column, much like above and for each token, calculate the tfidf
#list

punctuation = ["[","]",",",".","/",";","!","(",")","?"]

#list

nltk_stopwords = nltk.corpus.stopwords.words('russian')

nltk_stopwords.extend(punctuation)

def preprocess_text(x):
	x = str(x)
	x = str.lower(x)
	x = re.sub(',', '', x)
	x = [w for w in x.split() if w not in set(nltk_stopwords)]
	#print(type(x))
	return x

#succesfully removed punctuation
train["title"] = train["title"].apply(preprocess_text)
test["title"] = test["title"].apply(preprocess_text)

print(train['description'])

train["description"] = train["description"].apply(preprocess_text)
test["description"] = test["description"].apply(preprocess_text)

#train["title"].split(",")

title_vocabulary =[]
desc_vocabulary = []

train["title"].apply(lambda x: title_vocabulary.extend(x))
test["title"].apply(lambda x: title_vocabulary.extend(x))

train["description"].apply(lambda x: desc_vocabulary.extend(x))
test["description"].apply(lambda x: desc_vocabulary.extend(x))

#vocabulary = set(vocabulary)
#this is the number of counts per word

title_value_counts = pd.Series(title_vocabulary).value_counts()
desc_value_counts = pd.Series(desc_vocabulary).value_counts()

#need to find the number of counts per word in each row now
#, and divide by the corresponding value in value_counts
#print(value_counts)

#these are going to be new columns, only about 100 or so
title_cols_of_df = title_value_counts.index.values
desc_cols_of_df = desc_value_counts.index.values

#print(value_counts[value_counts>500000].index.values)

#new_df = pd.DataFrame(index = range(train.shape[0]), columns = new_cols_of_df)

title_unique_word_list = title_value_counts.index.values.tolist()
desc_unique_word_list = desc_value_counts.index.values.tolist()

title_dict = {word:index for index, word in enumerate(title_unique_word_list)}
desc_dict = {word:index for index, word in enumerate(desc_unique_word_list)}

#def create_word_list():
#	vocabulary = []
#	vocabulary.exten

###################################################################################################################

print("Counting number of documents that contain a certain word (once per row)")
'''
for row in range(train.shape[0]):

	new_dict.update({word: word+1 for word in new_dict.keys()})
	print(row)
	for i in new_dict:
		
		if i in train.ix[row,"title"]: new_dict[i] += 1
'''

def find_pad_length(x):
	pad_length = 0
	if len(x) > pad_length: pad_length = len(x)
	return pad_length	

title_pad_length = max(test["title"].apply(find_pad_length))
desc_pad_length = max(train["description"].apply(find_pad_length))

def dict_encoding(x, dictionary, new_pad_length):
	#pad_length = max(x.apply(find_pad_length))
	#print(pad_length	)
	row_list =[]
	for word in x:
		#print(new_dict[word])
		row_list.append(dictionary[word])
	while len(row_list) < new_pad_length: row_list.append(0)
	return row_list		


train["title"] = train["title"].apply(lambda x: dict_encoding(x,title_dict,title_pad_length))
test["title"] = test["title"].apply(lambda x: dict_encoding(x,title_dict,title_pad_length))

print(train['title'])
print(test['title'])

X_train_title = np.vstack(train["title"])
X_test_title = np.vstack(test["title"])


train["description"] = train["description"].apply(lambda x: dict_encoding(x,desc_dict,desc_pad_length))
test["description"] = test["description"].apply(lambda x: dict_encoding(x,desc_dict,desc_pad_length))

print(train['description'])
print(test['description'])
print(desc_pad_length)

X_train_desc = np.vstack(train["description"])
X_test_desc = np.vstack(test["description"])

print(X_train_desc.shape, X_test_desc.shape)

#X_train = hstack([(X_train_title), (X_train_desc)])
#X_test = hstack([(X_test_title), (X_test_desc)])

#print(X_train.shape,X_test.shape)
#X_train = np.hstack(X_train_title,X_train_desc)
#X_test = np.hstack(X_test_title,X_test_desc)
#X = X.reshape(X,(X.shape[0], len(pad_length)))
#X = pd.DataFrame(X)

#print(Y)
#for i in pad_length:
#	df.iloc[:,i] = train[]


#print(X.dtype)
###################################################################################################################
'''
print("Determining TF-IDF features for the most commonly used words and creating new columns with their TF-IDF values inserted")

for i in new_cols_of_df:

	for index,row in train.iterrows():

		if i in train.ix[index,'title']:
			new_df.ix[index,i] = train.ix[index,'title'].count(i)/(math.log(number_of_docs/new_dict[i]))

		else: 
			new_df.ix[index,i] = 0


X = X.toarray()
X_test = X_test.toarray()
print(desc_pad_length)
'''
test_shape = test.shape[0]

del train,test
print(desc_pad_length)
#for i in new_df.columns.values:
#TFIDF_features = csr_matrix((new_df.values))
#	X = hstack((X,np.array(new_df[i])[:,None]))
#what happened when i stacked a bunch of hidden layers was that the local accuracy went really high
#what is needed is for cross validation instead so that the fitting doesnt do this and you get a worse score on the public lb
max_len = 30
#model = Sequential()

#vocab_size means the number of words in the total vocabulary, the second argument is the number of docubements 
#or the number of inputs (rows), the input_length is the max length of the document 
#print(X)
#model.add(Embedding(len(unique_word_list), 30, input_shape=(1,)))

#Let the model know what inputs to expect as well as the shapes of the arrays.

title = Input(shape = (title_pad_length,), name = 'title')
emb_title = Embedding(len(title_unique_word_list) + 1, 30)(title)
#Flatten()
#desc = Input(shape = (1,), name = 'desc')
#emp_desc = Embedding(len(desc_unique_word_list) + 1, 30)(desc)
#Flatten()
desc = Input(shape = (desc_pad_length,), name="desc")
emb_desc = Embedding(len(desc_unique_word_list) + 1, 30)(desc)

#create 
X_train_layer = Input(shape = (X.shape[0],X.shape[1]), name="desc")
X_test_layer = Input(shape = (X_test.shape[0],X_test.shape[1]), name="desc")

axis=-1

#initialize embedding layers

rnn_layer1 = GRU(50)(emb_desc)

conc = concatenate([X_train_title,X_train_desc], axis = -1)

#Flatten ALL layers at once
X_train_layer_dense = Dense(48, activation='relu') #,kernel_regularizer=regularizers.l2(0.01)

main_layer = concatenate([rnn_layer1, Flatten()(emb_title), Flatten()(emb_desc), X_train_layer_dense])

#model = Model
#model.add(Dense(32,  kernel_regularizer=regularizers.l2(2e-5),
#                activity_regularizer=regularizers.l1(2e-5)))
#model.add(Dense(48, activation='relu')) #,kernel_regularizer=regularizers.l2(0.01)
#model.add(Dropout(0.2, noise_shape=None, seed=None))
#model.add(Dense(24, activation = "relu"))	#,kernel_regularizer=regularizers.l2(0.01)
#model.add(Dropout(0.2, noise_shape=None, seed=None))

main_l = Dropout(0.1)(Dense(512,activation='relu') (main_layer))

Y_layer = Dense(1, activation='sigmoid')(main_l)

model = Model(inputs = [title, desc,X_train_layer] ,output = Y_layer)

model.compile(loss='mean_squared_error', optimizer="adam", metrics=['mean_squared_error'])

X_train_title = np.array(X_train_title)
X_train_desc = np.array(X_train_desc)

model.fit([X_train_title,X_train_desc], Y, epochs=5, batch_size =5000)

#print(list(X_test))
predictions = model.predict([X_test_title,X_test_desc])

predictions = pd.Series(predictions.reshape(test_shape))
print(predictions)


item_id.reset_index(drop=True, inplace=True)
predictions.reset_index(drop=True, inplace=True)

df = pd.DataFrame(pd.concat([item_id, predictions], axis = 1)) #, columns = {"item_id","deal_probability"}
df.columns = ['item_id', "deal_probability"]

#df.drop(df.columns[0], axis = 1,inplace = True)

df.to_csv("C:\\Users\\Nick\\Desktop\\avito_competition\\predictions.csv", index = False)


'''
def tfidf(x,index):
	for i in new_cols_of_df:
		if i in x:
			new_df.ix[index,i] = x.count(i)/(math.log(number_of_docs/new_dict[i]))
		else: 
			new_df.ix[index,i] = 0
	return new_df
for i in 
train["title"].apply(tfidf)

print(new_df)

print(time.time() - start_time)

###################################################################################################################

#print(value_counts)

#need to split list entries for each row into different cols

#tokens = nltk.word_tokenize(train["title"][1])

#print(pd.Series(vocabulary).value_counts())
#for i in pd.unique(vocabulary):


#keras.sequential
'''