# 참고 : https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/

# 연관규칙분석
# 사전 설치
# C:\Users\Matthew\Downloads>pip install numpy-1.13.1+mkl-cp36-cp36m-win_amd64.whl
# C:\Users\Matthew\Downloads>pip install scipy-0.19.1-cp36-cp36m-win_amd64.whl
# C:\Users\Matthew\Downloads>pip install mlxtend
# C:\Users\Matthew\Downloads>pip install sklearn

# 필요한 라이브러리 로드
import pandas as pd

# 분석할 데이터 불러오기
titanic = pd.read_table('./data/titanic.txt')
titanic.shape # 읽어온 데이터 구조 확인
titanic.head() # 가장 앞 5개의 데이터를 출력
titanic.tail() # 가장 뒤 5개의 데이터를 출력

# Remove "Name" column
titanic_ar = titanic.iloc[:, 1:5]
titanic_ar.head()

# Age를 명목형 변수로 변경
c_idx = titanic_ar.loc[:,'Age'] < 20
a_idx = titanic_ar.loc[:,'Age'] >= 20
na_idx = titanic_ar.loc[:,'Age'].isnull()

titanic_ar.loc[c_idx,'Age'] = 'Child'
titanic_ar.loc[a_idx,'Age'] = 'Adult'
titanic_ar.loc[na_idx,'Age'] = 'Unknown'

titanic_ar.head(20) # 변경된 데이터 확인


survived_idx = titanic_ar.loc[:,'Survived'] == 1
dead_idx = titanic_ar.loc[:,'Survived'] == 0
titanic_ar.loc[survived_idx,'Survived'] = 'Survived'
titanic_ar.loc[dead_idx,'Survived'] = 'Dead'



from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import association_rules

dataset = titanic_ar.values.tolist()
oht = OnehotTransactions()
oht_ary = oht.fit(dataset).transform(dataset)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
df


frequent_itemsets = apriori(df, use_colnames=True,min_support=0.1)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
rules

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules


# 길이 정보를 갖는 열 추가
rules["antecedant_len"] = rules["antecedants"].apply(lambda x: len(x))
rules


# 길이가 2개 이상이고, Confidence가 0.75이상, lift가 1.2 이상인 Rule 출력
rules[ (rules['antecedant_len'] >= 2) &
       (rules['confidence'] > 0.75) &
       (rules['lift'] > 1.2) ]


# 죽은 경우에 대해서 Rule 분석
rules[ (rules['confidence'] > 0.75) &
       (rules['consequents'] == frozenset({'Dead'})) ]



import pandas as pd

transaction = open('./data/groceries.csv','r').readlines()
len(transaction)
gloceries = []
for line in transaction:
    line = line.replace('\n','').split(',')
    gloceries.append(line)
len(gloceries)


# pip install pytagcloud
# pip install pygame
# pip install simplejson
from collections import Counter
import pytagcloud
import itertools

nouns = list(itertools.chain(*gloceries))
count = Counter(nouns)
tag2 = count.most_common(100)
taglist = pytagcloud.make_tags(tag2, maxsize=80)
pytagcloud.create_tag_image(taglist, './wordcloud.jpg', size=(1024, 768), fontname='Coustard', rectangular=False)

# Plot histogram using matplotlib bar().
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.DataFrame.from_dict(count, orient='index')
df.columns = ['Count']
view = df.sort_values('Count',ascending=False)[0:30]
view.plot(kind='bar')


### Rule

oht = OnehotTransactions()
oht_ary = oht.fit(gloceries).transform(gloceries)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
df



frequent_itemsets = apriori(df, use_colnames=True,min_support=0.01)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0)
rules

# 길이 정보를 갖는 열 추가
rules["antecedant_len"] = rules["antecedants"].apply(lambda x: len(x))
rules

rules.sort_values('lift',ascending=False).head(10)

