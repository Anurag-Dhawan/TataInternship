from django.shortcuts import render,redirect
import joblib
import tensorflow
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.preprocessing.text import Tokenizer
from main.models import *
# Create your views here.
def home(request):
    if request.method=='POST':
        #print('hello')
        model = tensorflow.keras.models.load_model('finalized_model.h5')
        with open('saved_tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
         
        message = request.POST['text']
        name = request.POST['name']
        email = request.POST['email']
        company_name = request.POST['company_name']
        phone = request.POST['phone']

        seq = tokenizer.texts_to_sequences([message])
        #onehot_repr=[one_hot(words,voc_size)for words in new_complaint ] 
        padded = pad_sequences(seq, maxlen=250)
        pred = model.predict(padded)
        labels = ['1','2','3']
        print(pred,labels[np.argmax(pred)])
        res = int(labels[np.argmax(pred)])
        print(res)
        if res==1:
            a = negative(name = name, email = email,company_name=company_name,phone = phone , text = message)
            a.save()
        elif res==2:
            b = neutral(name = name, email = email,company_name=company_name,phone = phone , text = message)
            b.save()
        else :
            c = positive(name = name, email = email,company_name=company_name,phone = phone , text = message)
            c.save()
        return redirect('/')
    else:
        return render(request,'index.html')

    