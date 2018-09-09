import numpy as np
import pandas as pd
from pandas import DataFrame
import lightgbm as lgb
from sklearn import preprocessing
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from pandas import ExcelWriter
np.random.seed(1234*3)



from sklearn import ensemble
from sklearn.preprocessing import Imputer

result = pd.read_csv('./result_kor.csv', engine='python')
result['metric']=0
accuracy=[]
for i in range(16):
    if i in range(2,7):
        data_raw = pd.read_csv('./교통사망사고정보/Kor_Train_교통사망사고정보(12.1~17.6).csv', engine='python')

        test_kor = pd.read_csv('./test_kor.csv', engine='python')

        data_raw = data_raw.iloc[:, 3:22]
        data_raw = data_raw.drop(['사고유형', '법규위반_대분류', '당사자종별_1당', '당사자종별_1당'], axis=1)

        predict_name = data_raw.columns.tolist()
        data = pd.get_dummies(data_raw)
        data2 = pd.get_dummies(test_kor)
        data3 = data2.fillna(-1)
        data4 = pd.concat([data, data3], sort=False)

        data=data4.iloc[:-len(data3),]
        data_test_prep=data4.iloc[-len(data3):,]
        data_test_prep=data_test_prep.fillna(0)


        def full(column_name):
            a = result[result['열'] == column_name].index.tolist()
            b = result[result['열'] == column_name]
            c = (b['행'] - 2).tolist()
            predict_data = DataFrame(data_test_prep).reindex(c)
            predict_data = predict_data.drop(predict_name[i], axis=1)
            return a, predict_data


        ## 열 모델 만들기


        train, test = np.split(data.sample(frac=1), [ int(.8*len(data))])
        X_train=train.drop(predict_name[i], axis=1)
        Y_train=train[predict_name[i]]
        X_test=test.drop(predict_name[i], axis=1)
        Y_test=test[predict_name[i]]

        print("traing........")

        my_model = XGBRegressor(n_estimators=100, learning_rate=0.05)
        my_model.fit(X_train, Y_train, early_stopping_rounds=5,
                     eval_set=[(X_test, Y_test)], verbose=False)

        # make predictions

        predictions = my_model.predict(X_test)
        print("Mean Absolute Error : " + str(mean_squared_error(predictions, Y_test)))
        mse=str(mean_squared_error(predictions, Y_test))






        #열 예측하기
        Dict={0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M',13:'N', 14:'O', 15:'P'}
        index_1,k=full(Dict[i])
        print(k,X_train)
        e=pd.Series(my_model.predict(k))
        print(X_train)



        print(e)
        for i in range(len(e)):
            if e[i]>0:
                result.값[index_1[i]]=e[i]
            else:
                result.값[index_1[i]] = 0


        for i in range(len(e)):
            result.metric[index_1[i]] = 1.0/float(mse)
        print(result)



    elif i in range(16):
        data_raw = pd.read_csv('./교통사망사고정보/Kor_Train_교통사망사고정보(12.1~17.6).csv', engine='python')

        test_kor = pd.read_csv('./test_kor.csv', engine='python')

        data_raw=data_raw.iloc[:,3:22]
        data_raw=data_raw.drop(['사고유형','법규위반_대분류','당사자종별_1당','당사자종별_1당'],axis=1)


        data_concat = pd.concat([data_raw, test_kor], sort=False)

        predict_name=data_raw.columns.tolist()

        test_kor = test_kor.drop(predict_name[i],axis=1)




        #예측할 열만 인코딩


        le = preprocessing.LabelEncoder()
        le.fit(data_raw[predict_name[i]])
        category_size=len(le.classes_)
        data_raw[predict_name[i]]=le.transform(data_raw[predict_name[i]])




        """data = data_concat.drop(['발생지시군구'], axis=1)"""

        data = pd.get_dummies(data_raw)
        data2=pd.get_dummies(test_kor)
        data3=data2.fillna(-1)
        data4=pd.concat([data,data3],sort=False)

        data = data4.iloc[:-len(data3), ]
        data_test_prep=data4.iloc[-len(data3):,]
        data_test_prep=data_test_prep.fillna(0)

        """
        data4= data.sort_values([predict_name[i]],ascending=False)
    
        test_prep = test_kor.drop[predict_name[i]]"""



        def full(column_name):
            a=result[result['열'] == column_name].index.tolist()
            b=result[result['열'] == column_name]
            c=(b['행']-2).tolist()
            predict_data=DataFrame(data_test_prep).reindex(c)
            predict_data=predict_data.drop(predict_name[i], axis=1)
            return a, predict_data




        ## A 열 모델 만들기


        train, test = np.split(data.sample(frac=1), [ int(.8*len(data))])
        X_train=train.drop(predict_name[i], axis=1)
        Y_train=train[predict_name[i]]
        X_test=test.drop(predict_name[i], axis=1)
        Y_test=test[predict_name[i]]
        lgb_train=lgb.Dataset(X_train,Y_train)
        lgb_eval=lgb.Dataset(X_test, Y_test, reference=lgb_train)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': 'multi_error',
            'learning_rate': 0.1,
            'num_leaves': 255,
            'max_depth': 8,
            'min_child_samples': 100,
            'max_bin': 100,
            'subsample': 0.7,
            'subsample_freq': 1,
            'colsample_bytree': 0.7,
            'min_child_weight': 0,
            'subsample_for_bin': 200000,
            'min_split_gain': 0,
            'reg_alpha': 0,
            'reg_lambda': 0,
            # 'nthread': 8,
            'verbose': 0,
            'scale_pos_weight':99,
            'num_class':category_size
            }
        evals_results = {}

        print("Training the model...")

        lgb_model = lgb.train(params,
                         lgb_train,
                         valid_sets=[lgb_train, lgb_eval],
                         valid_names=['train','valid'],
                         evals_result=evals_results,
                        num_boost_round=1000,
                        early_stopping_rounds=15,
                        verbose_eval=True,
                        feval=None)



        lgb_model.reset_parameter({"num_threads":1})




        ## A 열 예측하기

        Dict={0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M',13:'N', 14:'O', 15:'P'}
        index_1,k=full(Dict[i])
        e=[]
        predictions = np.argmax(lgb_model.predict(X_test),axis=1)
        acc = str(accuracy_score(predictions, Y_test))
        for index, row in k.iterrows():
            e.append(le.inverse_transform(np.argmax(lgb_model.predict(row))))


        for i in range(len(e)):
            result.값[index_1[i]]=e[i]


        for i in range(len(e)):
            result.metric[index_1[i]] = acc
        print(result)
    else: continue


result.to_csv("BS_result_mat.csv", index=False, encoding='cp949')
"""writer = ExcelWriter('result_fill.xlsx')
result.to_excel(writer,'Sheet1')
writer.save()
"""