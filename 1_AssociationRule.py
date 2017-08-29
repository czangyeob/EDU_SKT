# 참고 : https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/

# 연관규칙분석
# 사전 설치
# C:\Users\Matthew\Downloads>pip install scipy-0.19.1-cp36-cp36m-win_amd64.whl
# Processing c:\users\matthew\downloads\scipy-0.19.1-cp36-cp36m-win_amd64.whl
# Requirement already satisfied: numpy>=1.8.2 in c:\python36\lib\site-packages (from scipy==0.19.1)
# Installing collected packages: scipy
# Successfully installed scipy-0.19.1
#
# C:\Users\Matthew\Downloads>pip install mlxtend
# Collecting mlxtend
#   Using cached mlxtend-0.7.0-py2.py3-none-any.whl
# Requirement already satisfied: scipy>=0.17 in c:\python36\lib\site-packages (from mlxtend)
# Requirement already satisfied: numpy>=1.10.4 in c:\python36\lib\site-packages (from mlxtend)
# Installing collected packages: mlxtend
# Successfully installed mlxtend-0.7.0
# C:\Users\Matthew\Downloads>pip install sklearn
# Collecting sklearn
#   Downloading sklearn-0.0.tar.gz
# Collecting scikit-learn (from sklearn)
#   Downloading scikit_learn-0.19.0-cp36-cp36m-win_amd64.whl (4.3MB)
#     100% |████████████████████████████████| 4.3MB 410kB/s
# Building wheels for collected packages: sklearn
#   Running setup.py bdist_wheel for sklearn ... done
#   Stored in directory: C:\Users\Matthew\AppData\Local\pip\Cache\wheels\d7\db\a3\1b8041ab0be63b5c96c503df8e757cf205c2848cf9ef55f85e
# Successfully built sklearn
# Installing collected packages: scikit-learn, sklearn
# Successfully installed scikit-learn-0.19.0 sklearn-0.0
# C:\Users\Matthew\Downloads>pip install numpy-1.13.1+mkl-cp36-cp36m-win_amd64.whl


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


transaction = open('./data/groceries.csv','r').readlines()
len(transaction)
gloceries = []
for line in transaction:
    line = line.replace('\n','').split(',')
    gloceries.append(line)
len(gloceries)

oht = OnehotTransactions()
oht_ary = oht.fit(gloceries).transform(gloceries)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
df



frequent_itemsets = apriori(df, use_colnames=True,min_support=0.01)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0)

# 길이 정보를 갖는 열 추가
rules["antecedant_len"] = rules["antecedants"].apply(lambda x: len(x))
rules

rules.sort_values('lift',ascending=False).head(10)

