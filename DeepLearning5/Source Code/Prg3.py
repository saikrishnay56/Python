#SPAM dataset sentiment analysis
import re
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


data = pd.read_csv('spam.csv', encoding='ISO-8859-1') #spam dataset

# Keeping only the neccessary columns
data = data[['v1', 'v2']]

# pre-processing
data['v2'] = data['v2'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x.lower()))


max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['v2'].values)
X = tokenizer.texts_to_sequences(data['v2'].values)
#print(X)
X = pad_sequences(X)
#print(X)
embed_dim = 128
lstm_out = 196

def createmodel():
    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['v1'])

y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#print(X_train.shape, Y_train.shape)
#print(X_test.shape, Y_test.shape)

# gridseachCV
model = KerasClassifier(build_fn=createmodel, verbose=0)
batch_size= [32,64]
epochs = [1,2]
param_grid= dict(batch_size=batch_size, epochs=epochs)

grid  = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result= grid.fit(X_train, Y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
