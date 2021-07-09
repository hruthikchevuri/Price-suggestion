# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [code]
df_train = pd.read_csv('../input/mercari/train.tsv', sep='\t')
df_test = pd.read_csv('../input/mercari/test.tsv', sep='\t')

# %% [code]
print('Train shape:',df_train.shape)
print('Test shape:',df_test.shape)

# %% [code]
df_train.head()

# %% [code]
df_train.describe

#plot for price distribution
plt.figure(figsize=(20, 15))
plt.hist(df_train['price'], bins=50, range=[0,250], label='price')
plt.title('Train "price" distribution')
plt.xlabel('Price')
plt.ylabel('No. of items')
plt.legend()
plt.show()

#plot for price distribution after log transformation
plt.figure(figsize=(8,5))

plt.hist(np.log1p(df_train['price']), bins=100)
plt.title('Train " log(price+1)" distribution', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

#snsplot for item condition vs price
sns.jointplot(x = df_train.item_condition_id, y = df_train['price'])

#plot for county of item_condition
count=df_train['item_condition_id'].value_counts()
plt.figure(figsize=(8,5))
sns.barplot(count.index[:10],count[:10])
plt.xlabel('Item_condition_id')
plt.ylabel('no. of items')
plt.show()

#plot for shipping
count=df_train['shipping'].value_counts()
plt.figure(figsize=(7,3))
plt.subplot(1,2,1)
sns.barplot(count.index,count)
plt.xlabel('Shipping')
plt.ylabel('Count')
plt.title('Shipping vs Count')
plt.subplot(1,2,2)
labels = ['0','1']
sizes = count
colors = ['green','red']
explode = (0.1, 0)  # explode 1st slice
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()


one_shipping=np.log(df_train.loc[df_train['shipping']==1,'price']+1)
zero_shipping=np.log(df_train.loc[df_train['shipping']==0,'price']+1)
sns.distplot(one_shipping,label='shipping = 1')
sns.distplot(zero_shipping,label='shipping = 0')
plt.title('distribution plot for shipping = 1 and shipping = 0')
plt.xlabel("log(price+1)")
plt.legend()
plt.show()


#plot for brand and price
top_50_expensive_brand=df_train.groupby(['brand_name'])['price'].mean().sort_values()[:15]
sns.set(rc={'figure.figsize':(8,5)})
top_50_expensive_brand.reset_index()
ax=sns.barplot(y='brand_name',x='price',data=top_50_expensive_brand.reset_index())
ax.set_xlabel('Average Price of products',fontsize=15)
ax.set_ylabel('Brand',fontsize=15)
plt.show()

#worclouds

from wordcloud import WordCloud

cloud = WordCloud(width=1440,height=1080).generate(" ".join(df_train['name'].astype(str)))
plt.figure(figsize=(18, 15))
plt.imshow(cloud)
plt.axis('off')
plt.show()


# %% [code]
#AS WE CAN SEE ABOVE WE HAVE 3 PARTS IN CATEGORY NAME. FIRST WE'LL SPLIT THEM.
def transform_category_name(category_name):
    try:
        main, sub1, sub2= category_name.split('/')
        return main, sub1, sub2
    except:
        return np.nan, np.nan, np.nan

df_train['category_main'], df_train['category_sub1'], df_train['category_sub2'] = zip(*df_train['category_name'].apply(transform_category_name))

# %% [code]
#NOW WE WILL FILL NULLS WITH FILLNA
df_train['category_main']=df_train['category_main'].fillna("No main Categ given")
df_train['category_sub1']=df_train['category_sub1'].fillna("No sub1 Categ given")
df_train['category_sub2']=df_train['category_sub2'].fillna("No sub2 Categ given")
df_train['brand_name']=df_train['brand_name'].fillna("No brand given")
#WE TRIED FILLING WITH MODE,MEAN,FFILL,BFILL BUT FILLING WITH NOT GIVEN RESPECTIVELY GAVE BEST RESULTS

# %% [code]
#import re

#def contracting(sentence):         #In this func few contractions are substituted instead of whole word
 #   sentence = re.sub(r"n\'t", " not", sentence)
 #   sentence = re.sub(r"\'ll", " will", sentence)
 #   sentence = re.sub(r"\'s", " is", sentence)
 #   sentence = re.sub(r"\'re", " are", sentence)
 #   sentence = re.sub(r"\'ve", " have", sentence)   #THIS FUNC DECREASED OUR ACCURACY SO WE COMMENTED IT 
 #   sentence = re.sub(r"\'d", " would", sentence)                                  
 #   sentence = re.sub(r"\'t", " not", sentence)
 #   sentence = re.sub(r"won't", "will not", sentence)
 #   sentence = re.sub(r"can't", "can not", sentence)
 #   return sentence
#
#
#
#def preprocess(data):             #General Preprocessing
 #   data = contracting(data)      #THIS FUNC DECREASED OUR ACCURACY SO WE COMMENTED IT 
 #   data = re.sub("[\-\\\n\t]", " ", data)  #Here we remove all \n, \t, - and \
 #   data = re.sub("[^A-Za-z0-9]", " ", data)  #Here we remove all the words except alphabets and numbers
 #   data = re.sub('\s\s+', ' ', str(data))  #Here we remove all the extra spaces
 #   data = data.lower() #This step is used to convert everything to lower case
 #   return data
#
#
#
#from nltk.corpus import stopwords
#import string
#from nltk.stem.porter import PorterStemmer
#
#
#porter = PorterStemmer()  #USING PORTER ALSO DECREASED OUR ACCURACY
#
#
#def punctuation_remover(sentence: str) -> str:
 #   return sentence.translate(str.maketrans('', '', string.punctuation))
#
#   
#WE USED PUNCTUATION REMOVER TO REMOVE PUNCTUATION BUT WE GOT BETTER RESULTS WITH INCLUDING PUNCTUATIONS.
#
#
#stop_words = stopwords.words('english')
#def stop_words_remover(k):
 #   k = k.lower()
 #   k = ' '.join([i for i in k.split(' ') if i not in stop_words])
 #   return k
#
#
#    
#INSTEAD OF USING STOP WORDS REMOVER WE KEPT STOPS WORDS IN TF-IDF THAT WAY WE REMOVED THIS FUNCTION
#TO USE THE INBUILT FACILITY AVAILABLE WITH TF-IDF

# %% [code]
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re

# %% [code]
def tokenizer(text):
    if text:
        result = re.findall('[a-z]{2,}', text.lower())
    else:
        result = []
    return result

# %% [code]
#IN THIS STEP WE DECLARED VARIABLES FOR TFIDF.
tfidf1 = TfidfVectorizer(max_features = 1000000,ngram_range=(1, 3),tokenizer=tokenizer,stop_words = stopwords.words('english'))
tfidf2 = TfidfVectorizer(max_features = 1000000,ngram_range=(1, 3),tokenizer=tokenizer,stop_words = stopwords.words('english'))
#BELOW ARE OUR FAILED TRIALS COMMENTED
#tfidf3 = CountVectorizer(max_features = 500000,ngram_range=(1, 3),tokenizer=tokenizer,min_df=10)
#tfidf4 = CountVectorizer(max_features = 500000,ngram_range=(1, 3),tokenizer=tokenizer,min_df=10)
#tfidf5 = CountVectorizer(max_features = 500000,ngram_range=(1, 3),tokenizer=tokenizer,min_df=10)

# %% [code]
#WE TRIED APPLYING TF-IDF FOR CATEGORY ALSO, BUT WE GOT BEST RESULT WITH THESE TWO ONLY
tfidf_description=tfidf1.fit_transform(df_train['item_description'].apply(str))
tfidf_name = tfidf2.fit_transform(df_train['name'])
#WE TRIED APPLYING COUNT VECTORIZER FOR NAME BUT TF-IDF WAS BETTER

# %% [code]
#WHEN COMPARED TO TF-IDF ONEHOT GAVE BETTER ACCURACY TO THE 3 PARTS OF CATEGORY
from sklearn.preprocessing import OneHotEncoder
one_encoder3 = OneHotEncoder(handle_unknown='ignore')
one_encoder4 = OneHotEncoder(handle_unknown='ignore')
one_encoder5 = OneHotEncoder(handle_unknown='ignore')
tfidf_main = one_encoder3.fit_transform(df_train[['category_main']])
tfidf_sub1 = one_encoder4.fit_transform(df_train[['category_sub1']])
tfidf_sub2 = one_encoder5.fit_transform(df_train[['category_sub2']])

# %% [code]
import scipy.sparse as spa

# %% [code]
a = spa.hstack((tfidf_description,tfidf_name,tfidf_main,tfidf_sub1,tfidf_sub2))

# %% [code]
one_encoder = OneHotEncoder(handle_unknown='ignore')
one_encoder1 = OneHotEncoder(handle_unknown='ignore')
one_encoder2 = OneHotEncoder(handle_unknown='ignore')

# %% [code]
#IN THIS STEP WE APPLIED ONEHOT FOR BRAND, SHHIPING, ITEM_CONDITION_ID
Brand = one_encoder.fit_transform(df_train[['brand_name']])
Shipping = one_encoder1.fit_transform(df_train[['shipping']])
NewId = one_encoder2.fit_transform(df_train[['item_condition_id']])

# %% [code]
X = spa.hstack((a,Brand,NewId,Shipping)).tocsr()

# %% [code]
Y = np.log(df_train['price']+1)
#HERE WE TOOK LOG TO PREVENT FROM GETTING SKEWED RESULTS

# %% [code]
#THIS IS JUST FOR CALCULATING ACCURACY
#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.05,random_state=0)

# %% [code]
#BELOW 5 STEPS YOU CAN SEE THE STEP BY STEP DEVELOPEMENT OF OUR HARDWORK ON MODELS
#WE STARTED WITH LINEAR REGRESSION AND AT LAST WE CONCLUDED WITH THE PAIR OF RIDGE AND LGBM.

# %% [code]
#from sklearn.model_selection import train_test_split#
#from sklearn.linear_model import LinearRegression
#reg = LinearRegression()
#reg.fit(X,Y)

# %% [code]
from sklearn.linear_model import Ridge
reg = Ridge(alpha = 1.5)
reg.fit(X,Y)

# %% [code]
import lightgbm as lgb
from lightgbm import LGBMRegressor
model1 =  LGBMRegressor(max_depth=10, n_estimators=2000)
model1.fit(X,Y)

# %% [code]
#import xgboost
#model2 = xgboost.XGBRegressor(random_state=42)
#model2.fit(X,Y)
#xgb DIDNT GAVE GOOD RESULTS

# %% [code]
#from sklearn.ensemble import RandomForestRegressor
#regr = RandomForestRegressor( max_features=500000, max_leaf_nodes=64,max_depth=8, random_state=0)
#regr.fit(X,Y)
#RANDOM FOREST ALSO DIDNT GIVE BETTER ACCURACY

# %% [code]
#NOW WE REPEATED SAME PROCESS FOR THE TEST DATA AS WELL

# %% [code]
df_test['category_main'], df_test['category_sub1'], df_test['category_sub2'] = zip(*df_test['category_name'].apply(transform_category_name))

# %% [code]
df_test['category_main']=df_test['category_main'].fillna("No main Categ given")
df_test['category_sub1']=df_test['category_sub1'].fillna("No sub1 Categ given")
df_test['category_sub2']=df_test['category_sub2'].fillna("No sub2 Categ given")
df_test['brand_name']=df_test['brand_name'].fillna("No brand given")
df_test['name'].fillna(df_test['name'].mode()[0],inplace=True)

# %% [code]
tfidf_t_description = tfidf1.transform(df_test['item_description'].apply(str))
tfidf_t_name =tfidf2.transform(df_test['name'])

# %% [code]
tfidf_t_main = one_encoder3.transform(df_test[['category_main']])
tfidf_t_sub1 = one_encoder4.transform(df_test[['category_sub1']])
tfidf_t_sub2 = one_encoder5.transform(df_test[['category_sub2']])

# %% [code]
b = spa.hstack((tfidf_t_description,tfidf_t_name,tfidf_t_main,tfidf_t_sub1,tfidf_t_sub2))

# %% [code]
TBrand = one_encoder.transform(df_test[['brand_name']])
TShipping = one_encoder1.transform(df_test[['shipping']])
TNewId = one_encoder2.transform(df_test[['item_condition_id']])

# %% [code]
Xt = spa.hstack((b,TBrand,TNewId,TShipping)).tocsr()

# %% [code]
pred_t = 0.5*reg.predict(Xt) + 0.5*model1.predict(Xt)
#
#
#BELOW ARE OUR FAILED RATIOS WHICH WE TRIED.
#pred_t = 0.2*reg.predict(Xt) + 0.8*model1.predict(Xt)
#pred_t = 0.6*reg.predict(Xt) + 0.4*model1.predict(Xt)
#
#
#NEXT WE TRIED TO COMBINE 4 MODELS BUT IT DIDNT IMPROVE OUR ACCURACY SO WE SETTLED WITH 2 MODELS
#
#
#pred_t = 0.7*reg.predict(Xt) + 0.1*model1.predict(Xt) + 0.1*regr.predict(Xt) + 0.1*model2.predict(Xt)
#pred_t = 0.8*reg.predict(Xt) + 0.03*model1.predict(Xt) + 0.15*regr.predict(Xt) + 0.02*model2.predict(Xt)

# %% [code]
#FOLLOWING STEPS ARE JUST TO CREATE A CSV FILE AS OUTPUT IN PROPER FORMAT.

# %% [code]
for i,n in enumerate(pred_t):
  if n<0:
    pred_t[i]=0

# %% [code]
df_test['price']=np.exp(pred_t)-1

# %% [code]
df_test[['id','price']].to_csv('final_output_with_combination15.csv', index=False)

# %% [code]
out=pd.read_csv('./final_output_with_combination15.csv')
out