#!/usr/bin/env python
# coding: utf-8

# # 제품 이상여부 판별 프로젝트
# 

# ## 1. 데이터 불러오기
# 

# ### 필수 라이브러리
# 

# In[84]:


import os
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from tqdm import tqdm


# ### 데이터 읽어오기
# 

# In[85]:


# 데이터 경로 설정 및 랜덤 시드 설정
ROOT_DIR = "data"
RANDOM_STATE = 110

# 데이터 로드
train_data = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))

# 데이터 확인
train_data.head()


# ### 언더 샘플링
# 

# 데이타 불균형을 해결하기 위해 언더 샘플링을 진행합니다.
# 

# In[86]:


# 언더 샘플링을 통한 데이터 불균형 처리
normal_ratio = 3.0  # 1.0 means 1:1 ratio

df_normal = train_data[train_data["target"] == "Normal"]
df_abnormal = train_data[train_data["target"] == "AbNormal"]

num_normal = len(df_normal)
num_abnormal = len(df_abnormal)
print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}")

df_normal = df_normal.sample(n=int(num_abnormal * normal_ratio), replace=False, random_state=RANDOM_STATE)
df_concat = pd.concat([df_normal, df_abnormal], axis=0).reset_index(drop=True)
df_concat.value_counts("target")


# ### 데이터 분할
# 

# In[87]:


# 데이터 분할 (train/validation)
df_train, df_val = train_test_split(
    df_concat,
    test_size=0.3,
    stratify=df_concat["target"],
    random_state=RANDOM_STATE,
)

# 데이터 통계 출력 함수 정의
def print_stats(df: pd.DataFrame):
    num_normal = len(df[df["target"] == "Normal"])
    num_abnormal = len(df[df["target"] == "AbNormal"])

    print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}" + f" ratio: {num_abnormal/num_normal}")

# 통계 출력
print(f"  \tAbnormal\tNormal")
print_stats(df_train)
print_stats(df_val)


# ## 3. 모델 학습
# 

# ### 모델 정의
# 

# In[88]:


model = RandomForestClassifier(random_state=RANDOM_STATE)


# ### 모델 학습
# 

# In[89]:


# 모델 정의
model = RandomForestClassifier(random_state=RANDOM_STATE)

# 피처 선택 및 전처리
features = []

for col in df_train.columns:
    try:
        df_train[col] = df_train[col].astype(int)
        features.append(col)
    except:
        continue

# 학습 데이터 준비
train_x = df_train[features]
train_y = df_train["target"]

# 모델 학습
model.fit(train_x, train_y)


# In[90]:


# 검증 데이터 준비
val_x = df_val[features]
val_y = df_val["target"]

# 예측 수행
val_pred = model.predict(val_x)

# 평가
f1 = f1_score(val_y, val_pred, pos_label='AbNormal')
print(f'Validation F1 Score: {f1:.4f}')


# ## 4. 제출하기
# 

# ### 테스트 데이터 예측
# 

# 테스트 데이터 불러오기
# 

# In[91]:


test_data = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))


# In[92]:


df_test_x = test_data[features]

for col in df_test_x.columns:
    try:
        df_test_x.loc[:, col] = df_test_x[col].astype(int)
    except:
        continue


# In[93]:


test_pred = model.predict(df_test_x)
test_pred


# ### 제출 파일 작성
# 

# In[94]:


# 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
df_sub = pd.read_csv("submission.csv")
df_sub["target"] = test_pred


# In[95]:


# 제출 파일 저장
df_sub.to_csv("submission.csv", index=False)



# **우측 상단의 제출 버튼을 클릭해 결과를 확인하세요**
# 

# In[ ]:





# In[ ]:




