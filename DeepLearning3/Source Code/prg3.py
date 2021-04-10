from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Flatten

categories = ['alt.atheism', 'sci.space']
df = fetch_20newsgroups(subset='train', shuffle=True, categories=categories)
sentences=df.data
y=df.target


le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)

max_review_len = max([len(s.split()) for s in sentences])
vocab_size = len(tokenizer.word_index)+1

X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)
X_train = pad_sequences(X_train_tokens,maxlen=max_review_len)
X_test = pad_sequences(X_test_tokens,maxlen=max_review_len)

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_review_len))
model.add(Flatten())
model.add(layers.Dense(300,input_dim=vocab_size, activation='relu'))
model.add(layers.Dense(3, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(test_acc)
