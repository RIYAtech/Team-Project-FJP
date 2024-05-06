from flask import Flask , render_template, url_for,request
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file 
import matplotlib.pyplot as plt
import string
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords  

app =Flask(__name__)
data=pd.read_csv('Dataset.csv')
data.drop(['job_id' , 'salary_range' , 'telecommuting' , 'has_company_logo' , 'has_questions'] , axis = 1,inplace = True)
data.fillna(' ',inplace=True)
def split(location):
  l =location.split(',')
  return l[0]

data['country'] =data.location.apply(split) 
data['text'] = data['title'] + ' '+ data['location'] + ' ' + data['department'] + ' ' + data['company_profile'] + ' '+ data['description'] + ' ' + data['requirements'] + ' ' + data['benefits'] + ' ' + data['industry']

data.drop(['title', 'location', 'department', 'company_profile', 'description',
       'requirements', 'benefits', 'employment_type', 'required_experience',
       'required_education', 'industry', 'function', 'country'],axis=1,inplace=True)

data.drop(data[data['fraudulent']==' '].index, inplace = True)

stop_words = set(stopwords.words("english"))
data['text'] = data['text'].apply(lambda x:x.lower())
data['text'] = data['text'].apply(lambda x:' '.join([word for word in x.split() if word not in(stop_words)]))

from sklearn.metrics import accuracy_score ,confusion_matrix ,classification_report
from sklearn.model_selection import train_test_split

X_train,X_test ,y_train,y_test = train_test_split(data.text, data.fraudulent ,test_size =0.20,random_state=20)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)


from sklearn.naive_bayes import  MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train.astype(int))
y_pred_nb = nb.predict(X_test_dtm)
y_test=y_test.astype('int')



def co_sim(t):
    l=[]
    for i in data['text']:
        l.append(i)
    s=" ".join(l)
    d= [t, s]
    print(t)
    count_vectorizer = CountVectorizer()
    vector_matrix = count_vectorizer.fit_transform(d)
    print(vector_matrix)
    cosine_similarity_matrix = cosine_similarity(vector_matrix)
    print(cosine_similarity_matrix)
    return cosine_similarity_matrix[0][1]
 


@app.route('/')

@app.route('/Home')
def Home():
    return render_template('Home.html')

@app.route('/About')
def about():
    return render_template('About.html')
    
@app.route('/Pred')
def Pred():
    return render_template('Pred.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        t=message
        a=co_sim(t)
        mypred=-1
        if a>=0.23:
            t=[t]
            v = vect.transform(t)
            mypred=nb.predict(v)
        elif a>0.10:
            mypred=1
        return render_template('Result.html',prediction=mypred)


if __name__=='__main__':
    app.run(port=4000)