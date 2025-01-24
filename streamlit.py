import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

train = pd.read_csv("/Users/le_fuu/Github_local/Phase_1/project_ml/train.csv")
test = pd.read_csv("/Users/le_fuu/Github_local/Phase_1/project_ml/test.csv")

# 50% и больше информации  отсутствует 
train.drop(['Id', 'LotFrontage', 'MasVnrType', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
       'MiscFeature','MasVnrArea'],axis=1, inplace=True)
test.drop(['Id', 'LotFrontage', 'MasVnrType', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
       'MiscFeature','MasVnrArea'],axis=1, inplace=True)

train['Alley'].fillna('NoAlley', inplace=True)
test['Alley'].fillna('NoAlley', inplace=True)
train["BsmtCond"].fillna("No Fireplace", inplace=True)
test["BsmtCond"].fillna("No Fireplace", inplace=True)
train["BsmtQual"].fillna("No Information",inplace=True)
test["BsmtQual"].fillna("No Information",inplace=True)
train["BsmtExposure"].fillna("No Exposure",inplace=True)
test["BsmtExposure"].fillna("No Exposure",inplace=True)
train["BsmtFinType1"].fillna("No Basement",inplace=True)
test["BsmtFinType1"].fillna("No Basement",inplace=True)
train["BsmtFinType2"].fillna("No Basement",inplace=True)
test["BsmtFinType2"].fillna("No Basement",inplace=True)
train['Electrical'].fillna('SBrkr', inplace=True)
test['Electrical'].fillna('SBrkr', inplace=True)

test.dropna(axis=0,inplace=True)

target_column = train["SalePrice"]

# Drop Target Feature
train.drop("SalePrice",axis=1,inplace=True)

train_num_feature = train.select_dtypes(include='number')
train_obj_feature = train.select_dtypes(include=['object', 'category'])

test_num_feature = test.select_dtypes(include='number')
test_obj_feature = test.select_dtypes(include=['object', 'category'])

OE = OrdinalEncoder(categories=[
    ['Fa','TA','Gd','Ex'],
    ['Po','Fa','TA','Gd','Ex'],
    ['No Information','Fa','TA','Gd','Ex'],
    ['No Fireplace','Po','Fa','TA','Gd'],
    ['No Exposure','No','Mn','Av','Gd'],
    ['No Basement','Unf','LwQ','BLQ','Rec','ALQ','GLQ'],
    ['No Basement','Unf','LwQ','BLQ','Rec','ALQ','GLQ'],
    ['Po','Fa','TA','Gd','Ex'],
    ['Fa','TA','Gd','Ex']
],dtype=np.int32)
# Ordinal Encoding on TrainData
train_obj_feature[['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual']] = OE.fit_transform(
    train_obj_feature[['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual']]
)

# Ordinal Encoding on TestData
test_obj_feature[['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual']] = OE.transform(
    test_obj_feature[['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual']]
)

ohe = OneHotEncoder(drop='first' , sparse_output=False , dtype=np.int32)

# OneHotEncoder on Train Data
train_data_ohe = ohe.fit_transform(train_obj_feature[['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope',
                               'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
                               'Exterior1st','Exterior2nd','Foundation','Heating','CentralAir','Electrical','Functional',
                               'PavedDrive','SaleCondition','SaleType']]
                 )

# OneHotEncoder on Train Data
test_data_ohe = ohe.transform(test_obj_feature[['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope',
                               'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
                               'Exterior1st','Exterior2nd','Foundation','Heating','CentralAir','Electrical','Functional',
                               'PavedDrive','SaleCondition','SaleType']]
                 )

ohe_columns = ohe.get_feature_names_out(['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope',
                               'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
                               'Exterior1st','Exterior2nd','Foundation','Heating','CentralAir','Electrical','Functional',
                               'PavedDrive','SaleCondition','SaleType'])

train_data_ohe = pd.DataFrame(columns=ohe_columns,data=train_data_ohe, index=train_obj_feature.index)
test_data_ohe = pd.DataFrame(columns=ohe_columns,data=test_data_ohe, index=test_obj_feature.index)

remaining_train_obj_feature = train_obj_feature.drop(['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig'
                  ,'LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                  'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','Foundation','Heating',
                  'CentralAir','Electrical','Functional','PavedDrive','SaleCondition','SaleType'], axis=1)
remaining_test_obj_feature = test_obj_feature.drop(['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig'
                  ,'LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                  'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','Foundation','Heating',
                  'CentralAir','Electrical','Functional','PavedDrive','SaleCondition','SaleType'], axis=1)

scaler = StandardScaler()

# Scaling Train Num Feature
scale_train_num_feature = scaler.fit_transform(train_num_feature)
train_num_feature = pd.DataFrame(data=scale_train_num_feature , columns=train_num_feature.columns, index=train_num_feature.index)

# Scaling Test Num Feature
scale_test_num_feature = scaler.transform(test_num_feature)
test_num_feature = pd.DataFrame(data=scale_test_num_feature , columns=test_num_feature.columns, index=test_num_feature.index)

train_data = pd.concat([train_num_feature, train_data_ohe,remaining_train_obj_feature],axis=1)
test_data = pd.concat([test_num_feature, test_data_ohe,remaining_test_obj_feature],axis=1)

X = train_data
y=target_column

x_train,x_test,y_train,y_test = train_test_split(X,y , test_size=0.2,random_state=42)

model_xgb = XGBRegressor(n_estimators=650, learning_rate=0.050365018932979656, max_depth=4)

model_xgb.fit(x_train, y_train)
# ___________________________________________________________________________________________________________________
st.title("Предсказание с помощью модели")

uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("Загруженные данные:")
    st.dataframe(df)
    id = df.shape[0]
    df.drop(['Id', 'LotFrontage', 'MasVnrType', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
       'MiscFeature','MasVnrArea'],axis=1, inplace=True)
    df['Alley'].fillna('NoAlley', inplace=True)
    df["BsmtCond"].fillna("No Fireplace", inplace=True)
    df["BsmtQual"].fillna("No Information",inplace=True)
    df["BsmtExposure"].fillna("No Exposure",inplace=True)
    df["BsmtFinType1"].fillna("No Basement",inplace=True)
    df["BsmtFinType2"].fillna("No Basement",inplace=True)
    df['Electrical'].fillna('SBrkr', inplace=True)
    df.dropna(axis=0,inplace=True)
    df_num_feature = df.select_dtypes(include='number')
    df_obj_feature = df.select_dtypes(include=['object', 'category'])
    df_obj_feature[['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual']] = OE.transform(
    df_obj_feature[['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual']])

    df_data_ohe = ohe.transform(df_obj_feature[['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope',
                               'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
                               'Exterior1st','Exterior2nd','Foundation','Heating','CentralAir','Electrical','Functional',
                               'PavedDrive','SaleCondition','SaleType']])
    df_data_ohe = pd.DataFrame(columns=ohe_columns,data=test_data_ohe, index=test_obj_feature.index)
    remaining_df_obj_feature = test_obj_feature.drop(['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig'
                  ,'LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                  'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','Foundation','Heating',
                  'CentralAir','Electrical','Functional','PavedDrive','SaleCondition','SaleType'], axis=1)
    
    df_num_feature = pd.DataFrame(data=scale_test_num_feature , columns=df_num_feature.columns, index=df_num_feature.index)

    df_data = pd.concat([test_num_feature, test_data_ohe,remaining_test_obj_feature],axis=1)

    st.write("Обработанные данные:")
    st.dataframe(df_data)

    st.write("Делаем предсказания...")
    predictions = model_xgb.predict(df_data)

    submission = pd.DataFrame({
    "Id": df_data.index,
    "SalePrice": predictions
    })

    st.write("Итоговый DataFrame в формате submission:")
    st.dataframe(submission)

    st.write("Скачайте файл для отправки:")
    submission_file = "result_predict.csv"
    submission.to_csv(submission_file, index=False)

    st.download_button(
        label="Скачать result_predict.csv",
        data=submission.to_csv(index=False),
        file_name=submission_file,
        mime="text/csv")