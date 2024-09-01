#!/usr/bin/env python
# coding: utf-8

# # 제품 이상여부 판별 프로젝트
# 

# ## 1. 데이터 불러오기
# 

# ### 필수 라이브러리
# 

# In[108]:


#!/usr/bin/env python
# coding: utf-8

# # 제품 이상여부 판별 프로젝트 (code.ipynb)

# ## 1. 데이터 불러오기 및 필수 라이브러리 설정
get_ipython().system('pip install imbalanced-learn')
# ### 필수 라이브러리
import os
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE  # SMOTE 사용
from sklearn.preprocessing import StandardScaler  # 스케일링
from tqdm import tqdm

# ### 데이터 읽어오기 (data/train.csv)

# 데이터 경로 설정 및 랜덤 시드 설정
RANDOM_STATE = 110

# 데이터 로드
train_data = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))

# 데이터 확인
train_data.head()



# ### 데이터 읽어오기
# 

# In[109]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# 데이터 불러오기
train_data = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))

# 모든 값이 NaN인 열 제거
train_data = train_data.dropna(axis=1, how='all')

# 타겟과 피처 분리
train_x = train_data.drop("target", axis=1)
train_y = train_data["target"]

# 숫자형 피처와 비숫자형 피처를 분리
numeric_features = train_x.select_dtypes(include=[np.number]).columns
categorical_features = train_x.select_dtypes(exclude=[np.number]).columns

# 숫자형 피처에 대해서만 결측값 처리 (평균으로 대체)
imputer = SimpleImputer(strategy='mean')
train_x_numeric_imputed = pd.DataFrame(imputer.fit_transform(train_x[numeric_features]), 
                                       columns=numeric_features)

# 비숫자형 피처에 대해서는 가장 빈번한 값으로 결측값 처리
imputer_cat = SimpleImputer(strategy='most_frequent')
train_x_categorical_imputed = pd.DataFrame(imputer_cat.fit_transform(train_x[categorical_features]), 
                                           columns=categorical_features)

# 비숫자형 피처를 Label Encoding으로 변환
label_encoders = {}
for column in train_x_categorical_imputed.columns:
    le = LabelEncoder()
    train_x_categorical_imputed[column] = le.fit_transform(train_x_categorical_imputed[column])
    label_encoders[column] = le

# 숫자형 피처와 인코딩된 비숫자형 피처를 병합
train_x_imputed = pd.concat([train_x_numeric_imputed, train_x_categorical_imputed], axis=1)

# 열 개수 확인
print(f"Original columns: {train_data.shape[1]-1}, Imputed columns: {train_x_imputed.shape[1]}")

# 이후 단계로 진행


# ### 언더 샘플링
# 

# 데이타 불균형을 해결하기 위해 언더 샘플링을 진행합니다.
# 

# In[110]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# 데이터 불러오기
train_data = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))

# 타겟과 피처 분리
train_x = train_data.drop("target", axis=1)
train_y = train_data["target"]

# 모든 값이 NaN인 열 제거
train_x = train_x.dropna(axis=1, how='all')

# 숫자형 피처와 비숫자형 피처를 분리
numeric_features = train_x.select_dtypes(include=[np.number]).columns
categorical_features = train_x.select_dtypes(exclude=[np.number]).columns

# 숫자형 피처에 대해서만 결측값 처리 (평균으로 대체)
imputer = SimpleImputer(strategy='mean')
train_x_numeric_imputed = pd.DataFrame(imputer.fit_transform(train_x[numeric_features]), 
                                       columns=numeric_features)

# 비숫자형 피처에 대해서는 가장 빈번한 값으로 결측값 처리
imputer_cat = SimpleImputer(strategy='most_frequent')
train_x_categorical_imputed = pd.DataFrame(imputer_cat.fit_transform(train_x[categorical_features]), 
                                           columns=categorical_features)

# 비숫자형 피처를 Label Encoding으로 변환
label_encoders = {}
for column in train_x_categorical_imputed.columns:
    le = LabelEncoder()
    train_x_categorical_imputed[column] = le.fit_transform(train_x_categorical_imputed[column])
    label_encoders[column] = le

# 숫자형 피처와 인코딩된 비숫자형 피처를 병합
train_x_imputed = pd.concat([train_x_numeric_imputed, train_x_categorical_imputed], axis=1)

# SMOTE 적용
smote = SMOTE(random_state=RANDOM_STATE)
train_x_resampled, train_y_resampled = smote.fit_resample(train_x_imputed, train_y)

# 이후 train_x_resampled과 train_y_resampled을 사용하여 학습 진행


# ### 데이터 분할
# 

# In[111]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# 이 부분은 이미 처리된 데이터라고 가정합니다.
# df_train, df_val, y_train, y_val = train_test_split(...)

# train_x_resampled과 train_y_resampled이 이미 SMOTE를 통해 생성된 데이터입니다.
df_train = train_x_resampled
y_train = train_y_resampled

# F1 스코어 계산 시 양성 클래스로 'AbNormal'을 지정
f1_scorer = make_scorer(f1_score, pos_label='AbNormal')

# 모델 학습 및 하이퍼파라미터 튜닝
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

model = RandomForestClassifier(random_state=RANDOM_STATE)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           scoring=f1_scorer, cv=3, n_jobs=-1, verbose=2)

grid_search.fit(df_train, y_train)

best_model = grid_search.best_estimator_

# 이후 best_model을 사용하여 검증 및 예측 작업을 진행합니다.


# ## 3. 모델 학습
# 

# ### 모델 정의
# 

# In[112]:


# ## 4. 모델 성능 평가

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# train_x_resampled과 train_y_resampled이 이미 SMOTE를 통해 생성된 데이터라고 가정합니다.
df_train, df_val, y_train, y_val = train_test_split(
    train_x_resampled, train_y_resampled, test_size=0.3, random_state=RANDOM_STATE, stratify=train_y_resampled
)

# 모델 학습 및 하이퍼파라미터 튜닝이 완료되었다고 가정하고 best_model이 있습니다.

# 검증 데이터로 예측 수행
val_pred = best_model.predict(df_val)

# 평가 (F1 스코어 계산)
f1 = f1_score(y_val, val_pred, pos_label='AbNormal')
print(f'Validation F1 Score: {f1:.4f}')



# ### 모델 학습
# 

# In[131]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os

# 데이터 경로 설정 및 랜덤 시드 설정
RANDOM_STATE = 110
ROOT_DIR = "data"  # 데이터 파일이 있는 디렉토리

# 학습 데이터 로드
train_data = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))

# 테스트 데이터 로드
test_data = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))
test_x = test_data.copy()

# 학습 데이터의 모든 피처를 추출
train_features = train_data.columns.drop('target')  # 'target'을 제외한 모든 피처
numeric_features = train_data.select_dtypes(include=[np.number]).columns

# 테스트 데이터에 학습 데이터에서 사용된 피처가 없다면 추가하고 NaN으로 채우기
for col in train_features:
    if col not in test_x.columns:
        test_x[col] = np.nan

# 테스트 데이터에서 학습 데이터에 없는 피처 제거
test_x = test_x[train_features]

# 학습 데이터와 테스트 데이터의 피처 순서를 일치시키기 위해 동일한 순서로 정렬
test_x = test_x[train_features]

# 숫자형 피처와 비숫자형 피처를 분리
numeric_test_features = test_x.select_dtypes(include=[np.number]).columns
categorical_test_features = test_x.select_dtypes(exclude=[np.number]).columns

# 학습 데이터에서 이미 학습된 imputer를 사용하여 테스트 데이터의 숫자형 피처 결측값 처리
imputer = SimpleImputer(strategy='mean')
train_x_numeric_imputed = imputer.fit_transform(train_data[numeric_features])
test_x_numeric_imputed = imputer.transform(test_x[numeric_test_features])

# 비숫자형 피처에 대해서 결측값 처리 (가장 빈번한 값으로 대체)
imputer_cat = SimpleImputer(strategy='most_frequent')
train_x_categorical_imputed = imputer_cat.fit_transform(train_data[categorical_test_features])
test_x_categorical_imputed = imputer_cat.transform(test_x[categorical_test_features])

# 비숫자형 피처를 Label Encoding으로 변환
label_encoders = {}
for column in categorical_test_features:
    le = LabelEncoder()
    train_x_categorical_imputed[:, categorical_test_features.get_loc(column)] = le.fit_transform(train_data[column].fillna('Missing'))
    
    # 학습 데이터에 없는 새로운 카테고리를 처리하기 위해 예외 처리 추가
    test_x_categorical_imputed[:, categorical_test_features.get_loc(column)] = test_x[column].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else len(le.classes_)
    )
    label_encoders[column] = le

# 숫자형 피처와 인코딩된 비숫자형 피처를 병합
train_x_imputed = np.hstack((train_x_numeric_imputed, train_x_categorical_imputed))
test_x_imputed = np.hstack((test_x_numeric_imputed, test_x_categorical_imputed))

# 최종 모델로 학습
best_model = RandomForestClassifier(random_state=RANDOM_STATE)
train_y = train_data['target']
best_model.fit(train_x_imputed, train_y)

# 테스트 데이터 예측
test_pred = best_model.predict(test_x_imputed)

# 결과를 submission.csv 파일로 저장
submission = pd.DataFrame({
    'Set ID': test_data['Set ID'],  # test.csv 파일의 'Set ID' 열을 사용
    'target': test_pred             # 예측된 결과를 'target' 열에 할당
})

# 'target' 열에 저장된 결과를 정상과 비정상으로 변환 (예: 1 -> 'AbNormal', 0 -> 'Normal')
submission['target'] = submission['target'].apply(lambda x: 'AbNormal' if x == 1 else 'Normal')

# 현재 디렉터리에 submission.csv 파일로 저장
submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")


# In[ ]:





# ## 4. 제출하기
# 

# ### 테스트 데이터 예측
# 

# 테스트 데이터 불러오기
# 

# ### 제출 파일 작성
# 

# **우측 상단의 제출 버튼을 클릭해 결과를 확인하세요**
# 

# In[ ]:





# In[ ]:




