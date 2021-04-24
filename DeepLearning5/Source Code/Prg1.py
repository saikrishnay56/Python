import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
import numpy as np

data = pd.read_csv('Sentiment.csv')

# Keeping only the neccessary columns
data = data[['text', 'sentiment']]

# Pre-processing
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

# tokeninzation  
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)

# padding to ensure uniformity in shape
X = pad_sequences(X)

# y values from categorical to integer coded
labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['sentiment'])
y = to_categorical(integer_encoded)

# define model
embed_dim = 128
lstm_out = 196


def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# print(model.summary())

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# train model 
batch_size = 32
model = createmodel()
model.fit(X_train, Y_train, epochs=1, batch_size=batch_size, verbose=2)

# save to disk
model1_json = model.to_json()  # only architecture
with open('lstm.json', 'w') as json_file:
    json_file.write(model1_json)
model.save_weights('lstm.h5')  # weights

# load architecture and weights
model = model_from_json(model1_json)
model.load_weights('lstm.h5')

# predict over test data
twt = ['A lot of good things are happening. We are respected again throughout the world, and that\'s a great thing']
# preprocessing same as above
twt = twt[0].lower()
twt = re.sub('[^a-zA-z0-9\s]', '', twt)

# tokenization
twt = tokenizer.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)

# make prediction based on trained model 
sentiment = model.predict(twt, batch_size=1, verbose=2)[0]

# 0 is negative, 1 is neutral and 2 is positive 
if (np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 2):
    print("positive")
else:
    print('neutral')
