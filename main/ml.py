import pandas as pd
import numpy as np
import string
import re
import nltk
import sklearn
from tensorflow.keras.layers import Embedding,Dropout,SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow import keras
import pickle

df = pd.read_csv(r'C:\Users\ANURAG DHAWAN\Desktop\PROJECT\Datasets\amazon_reviews.csv')

df = df.drop(['Unnamed: 0','Customer Name','Title'],axis=1)
pd.set_option('display.max_colwidth',100)

nltk.download('stopwords')

def data_clean(text):
    text=str(text)
    translator= str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(translator)
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  
                           u"\U0001F300-\U0001F5FF"  
                           u"\U0001F680-\U0001F6FF"  
                           u"\U0001F1E0-\U0001F1FF"  
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = text.strip()
    tokens = re.split("\W+",text)
    stopwords = nltk.corpus.stopwords.words('english')
    text = [word.lower() for word in tokens if word not in stopwords]
    #text = [i for i in text if not i.isdigit()]
    return text

df['clean_review']= df['Review'].apply(lambda x: data_clean(x))

wn = nltk.WordNetLemmatizer()
def lemmatization(tokens):
    text = [wn.lemmatize(word) for word in tokens]
    return ' '.join(word for word in text)

df['lemmatized'] = df['clean_review'].apply(lambda x: lemmatization(x))

def feedback(text):
    if(int(text[0])>3):
        return 3
    if(int(text[0])<3):
        return 1
    else:
        return 2
df['feedback'] = df['Rating'].apply(lambda x : feedback(x))    

x = df.lemmatized
y=df.feedback
Y = pd.get_dummies(df['feedback']).values

corpus=[]
for sentence in x:
    corpus.append(sentence)

type(corpus)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(corpus, Y, test_size=0.1, random_state=42 , stratify = Y)
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=50000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True ,oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(X_train)
training_padded = pad_sequences(training_sequences, maxlen=250)

testing_sequences = tokenizer.texts_to_sequences(X_test)
testing_padded = pad_sequences(testing_sequences, maxlen=250)
with open('saved_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

voc_size = 50000
sent_length=250
embedding_vector_features=100
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100,dropout=0.7, recurrent_dropout=0.7))
model.add(Dense(32,activation='sigmoid'))
model.add(Dense(3,activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.0002)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model_hist=model.fit(training_padded,y_train,epochs=20, validation_data=(testing_padded, y_test) ,batch_size=128)
print(model.evaluate(testing_padded,y_test))

#filename = 'finalized_model.pkl'
#joblib.dump(model,filename)
model.save('finalized_model.h5')