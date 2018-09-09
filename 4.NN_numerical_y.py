
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import keras
from pandas import DataFrame
from keras import models
from keras.layers import Conv1D, Concatenate, merge
from keras.models import load_model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import keras,pickle
from keras.layers import Activation, Dense, InputLayer, MaxPool1D, Flatten, Dropout, AvgPool1D, Input, concatenate, Concatenate, Add, Reshape
from keras.utils import np_utils
from keras.utils import np_utils
from keras.layers.merge import Concatenate
from keras.models import Model
np.random.seed(10)
from tensorflow import set_random_seed
set_random_seed(15)
from pprint import pprint
from keras.layers.embeddings import Embedding
from numpy import newaxis
from keras import backend as K
# from keras.utils import plot_model


result = pd.read_csv('./DL_result_mat.csv', engine ='python')
# result['metric_acc'] = 0



for model_num in range(5):











    kor_dum_sim = pd.read_csv('./교통사망사고정보/Kor_Train_교통사망사고정보(12.1~17.6).csv', engine='python')

    kor_dum_sim
    data_categorical = kor_dum_sim[[ '주야', '요일','발생지시도','발생지시군구','사고유형_대분류','사고유형_중분류','법규위반','도로형태_대분류','도로형태','당사자종별_1당_대분류','당사자종별_2당_대분류']]
    data_numerical = kor_dum_sim[[ '사망자수', '사상자수', '중상자수', '경상자수','부상신고자수']]

    print(data_categorical.columns.shape)
    print(data_numerical.columns.shape)
    print(kor_dum_sim.columns.shape)
    target_switch = data_numerical.columns[model_num]
    early_stopping = EarlyStopping(monitor='val_loss', patience=5) # Early stopping on val loss - not used
    #patience 5번동안 개선이 안되면 멈춤.


    # In[3]:


    #y로 둘 변수 선택
    #********************************y 타겟 (스위치)*************************************************************************************************
    y = data_numerical[[target_switch]] # 얘를 바꾸면 데이터가 다 바뀜.
    data_numerical = data_numerical.drop(y.columns[0], 1)

    # print(y)


    # In[4]:


    len(data_categorical.columns)
    list_categorical_without_y = []
    for i in range(len(data_categorical.columns)):
        list_categorical_without_y.append(data_categorical.columns[i])
    print(list_categorical_without_y)

    list_numerical_without_y = []
    for i in range(len(data_numerical.columns)):
        list_numerical_without_y.append(data_numerical.columns[i])
    print(list_numerical_without_y)


    # In[5]:




    y_target_class_num = len(pd.unique(y.iloc[:,0]))
    pprint(y_target_class_num)
    pprint(pd.unique(y.iloc[:,0]))


    # In[6]:


    #범주형 embedding_CNN

    dictionary_categorical = {c: list(data_categorical[c].unique()) for c in data_categorical.columns }
    print(dictionary_categorical)

    print('------------------')
    embed_cols = []
    for c in dictionary_categorical:
        if len(dictionary_categorical[c])>1:
            embed_cols.append(c)
            print(c + ': %d values' % len(dictionary_categorical[c])) #look at value counts to know the embedding dimensions
    print('------------------')
    print("embed_cols: ", embed_cols)
    print("embed_cols 개수 : ", len(embed_cols))


    # In[7]:



    def categorical_preproc_x(data):
        input_list = []

        # the cols to be embedded: rescaling to range [0, # values)
        for c in embed_cols:
            raw_vals = np.unique(data[c])
            val_map = {}
            for i in range(len(raw_vals)):
                val_map[raw_vals[i]] = i
            input_list.append(data[c].map(val_map).values)

        # the rest of the columns
        other_cols = [c for c in data.columns if (not c in embed_cols)]
        input_list.append(data[other_cols].values)
        input_list= input_list[0:len(embed_cols)] # 마지막 리스트에 빈거 생겨서 처리

        return input_list


    # In[8]:


    def categorical_mapping_for_testing_data_x(data, testing_data):
        testing_input_list = []

        for c in embed_cols:
            raw_vals = np.unique(data[c])
            val_map = {}
            for i in range(len(raw_vals)):
                val_map[raw_vals[i]] = i
    #         print(testing_data[c].map(val_map).values)
            testing_input_list.append(testing_data[c].map(val_map).values)

        return testing_input_list



    # In[9]:


    # 여러 모델에 넣을 거 대비해서 embed size 딕셔너리 만들기.
    dictionary_embed_size = {}
    dictionary_vocab_size = {}

    for i in embed_cols :

        no_of_unique_cat  = data_categorical[ i ].nunique()
        embedding_size = min(np.ceil((no_of_unique_cat)/2), 50 )
        embedding_size = int(embedding_size)
        if embedding_size == 1:
            embedding_size = 2
        vocab  = no_of_unique_cat+1

        dictionary_embed_size[ i ] = embedding_size
        dictionary_vocab_size[ i ] = vocab

    list_embed_size = list(dictionary_embed_size.values())
    list_vocab_size = list(dictionary_vocab_size.values())


    # In[10]:


    data_preproc_categorical = categorical_preproc_x(data_categorical)
    data_preproc_categorical


    # In[11]:


    def add_one_axis(data):
    #     data = np.array(data)
    #     data = data.T
        data = data[:,:, newaxis]
        return data



    # In[12]:


    (list_embed_size)


    # In[13]:


    #범주형 임베딩 레이어 작성
    # 17개 중 예측 할 변수를 제외하고 쌓기.




    input_1 = Input(shape = (1, ))
    embed_1 = Embedding(list_vocab_size[0], list_embed_size[0])(input_1)
    embed_1 = Reshape(target_shape=(list_embed_size[0], ))(embed_1)

    input_2 = Input(shape = (1, ))
    embed_2 = Embedding(list_vocab_size[1], list_embed_size[1])(input_2)
    embed_2 = Reshape(target_shape=(list_embed_size[1], ))(embed_2)

    input_3 = Input(shape = (1, ))
    embed_3 = Embedding(list_vocab_size[2], list_embed_size[2])(input_3)
    embed_3 = Reshape(target_shape=(list_embed_size[2], ))(embed_3)

    input_4 = Input(shape = (1, ))
    embed_4 = Embedding(list_vocab_size[3], list_embed_size[3])(input_4)
    embed_4 = Reshape(target_shape=(list_embed_size[3], ))(embed_4)

    input_5 = Input(shape = (1, ))
    embed_5 = Embedding(list_vocab_size[4], list_embed_size[4])(input_5)
    embed_5 = Reshape(target_shape=(list_embed_size[4], ))(embed_5)

    input_6 = Input(shape = (1, ))
    embed_6 = Embedding(list_vocab_size[5], list_embed_size[5])(input_6)
    embed_6 = Reshape(target_shape=(list_embed_size[5], ))(embed_6)

    input_7 = Input(shape = (1, ))
    embed_7 = Embedding(list_vocab_size[6], list_embed_size[6])(input_7)
    embed_7 = Reshape(target_shape=(list_embed_size[6], ))(embed_7)

    input_8 = Input(shape = (1, ))
    embed_8 = Embedding(list_vocab_size[7], list_embed_size[7])(input_8)
    embed_8 = Reshape(target_shape=(list_embed_size[7], ))(embed_8)

    input_9 = Input(shape = (1, ))
    embed_9 = Embedding(list_vocab_size[8], list_embed_size[8])(input_9)
    embed_9 = Reshape(target_shape=(list_embed_size[8], ))(embed_9)

    input_10 = Input(shape = (1, ))
    embed_10 = Embedding(list_vocab_size[9], list_embed_size[9])(input_10)
    embed_10 = Reshape(target_shape=(list_embed_size[9], ))(embed_10)

    ##범주에서 y가 없으면 11ayer도 실행시키기#### ************************************************************************************************(스위치)

    input_11 = Input(shape = (1, ))
    embed_11= Embedding(list_vocab_size[10], list_embed_size[10])(input_11)
    embed_11 = Reshape(target_shape=(list_embed_size[10], ))(embed_11)




    my_inputs = [input_1,input_2,input_3,input_4,input_5,input_6,input_7,input_8,input_9,input_10,input_11]
    #************************************************************************************************************************(스위치)



    my_embeddings = [embed_1,embed_2,embed_3,embed_4,embed_5,embed_6,embed_7,embed_8,embed_9,embed_10,embed_11]
    # ***********************************************************************************************************************(스위치)

    pprint( my_embeddings)
    print('**************************************************')
    pprint( my_inputs)
    pprint( type(my_inputs))


    # In[14]:


    #범주형 concat 후 CNN에 넣은 layer
    # embed_output = keras.layers.concatenate(my_embeddings)
    embed_output = Concatenate(axis = -1)(my_embeddings)
    print(embed_output)
    # embed_output = add_one_axis(embed_output)
    # print(embed_output)



    # conv_c1 = Conv1D(10, strides=1, kernel_size=(3), activation='relu', padding='same', kernel_initializer='he_normal')(embed_output)
    # print(conv_c1)
    # conv_c1 = Flatten()(conv_c1)
    # print(conv_c1)




    # In[15]:


    #수치형 input /CNN


    type(data_numerical)
    data_array_numerical = np.array(data_numerical)
    data_array_numerical



    my_input_numerical = Input( shape = (4,))
    my_input_numerical_reshape = Reshape(target_shape=( 4, ))(my_input_numerical)


    # my_input_numerical = add_one_axis(my_input_numerical) # 차원 추가 CNN용


    # conv_n1 = Conv1D(20, strides= 1, kernel_size=(3), activation='relu', padding= 'same', kernel_initializer='he_normal')(input_numerical)


    # In[16]:


    # 범주형 수치형 Concat
    print(my_input_numerical.shape)
    print('***********************************')
    my_embeddings.append(my_input_numerical_reshape)
    pprint(my_embeddings)

    print('***********************************')
    my_output = Concatenate(axis = -1)(my_embeddings)
    print(my_output.shape)


    # In[17]:


    my_output


    # In[18]:


    my_inputs.append(my_input_numerical)
    print(my_inputs)
    print(my_input_numerical)


    # In[19]:


    # 최종 모델

    output_model = Dense(240)(my_output)
    print(output_model)
    output_model = Activation('relu')(output_model)
    print(output_model)
    output_model = Dense(120)(output_model)
    print(output_model)
    output_model = Dense(60)(output_model)
    print(output_model)
    output_model = Dense(30)(output_model)
    print(output_model)

    output_model = Activation('relu')(output_model)
    print(output_model)
    output_model = Dense(1)(output_model)
    print(output_model)
    ## *****y_target_class_num******************************************************************************************************************(스위치)


    my_model = Model(inputs= my_inputs, outputs = output_model)
    my_model
    my_model.compile( loss = keras.losses.MSE, optimizer=keras.optimizers.Adam(lr=0.0001),metrics=['mse'])


    # In[20]:


    data_preproc_categorical.append(data_array_numerical)
    data_total =data_preproc_categorical
    print(data_total)


    # In[21]:


    # y_train에는 하나의 숫자값이 들어가야됨.

    if not os.path.isdir('./log'):
        os.mkdir('./log')
    if not os.path.isdir('./log/model'):
        os.mkdir('./log/model')
    # filepath = "./log/numerical_y_model_%s.hdf5"%model_num
    filepath = "./log/numerical_y_model_%s.hdf5"%1

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,save_best_only=True, mode='min')

    my_model.fit(data_total, y, epochs=20, validation_split = 0.25,callbacks=[early_stopping, checkpoint])
    ## *********************validation 비율*******************************************************************(스위치)


    # In[26]:


    mse = my_model.evaluate(data_total, y)[1]


    # In[ ]:


    my_testing_data = pd.read_csv('./ML_result_input.csv', engine='python')

    def full(column_name):
        a = result[result['열'] == column_name].index.tolist()
        b = result[result['열'] == column_name]
        c = (b['행'] - 2).tolist()
        predict_data = DataFrame(my_testing_data).reindex(c)
        return a, predict_data

    Dict={0:'C', 1:'D', 2:'E', 3:'F', 4:'G'}

    a,predict_data=full(Dict[model_num])

    testing_data_categorical = predict_data[list_categorical_without_y]
    testing_data_numerical = predict_data[list_numerical_without_y]


    def testing_data_prepoc(testing_data_categorical, testing_data_numerical):

        testing_data_categorical = categorical_mapping_for_testing_data_x(data_categorical, testing_data_categorical)
        testing_data_numerical = np.array(testing_data_numerical)

        testing_data_categorical.append(testing_data_numerical)

        testing_data_total = testing_data_categorical
        return testing_data_total

    testing_data_total = testing_data_prepoc(testing_data_categorical, testing_data_numerical)



    print(my_model.predict(testing_data_total))
    print("mse 입니다아아아아아아아아아아아아아아아아아아아아아아앙", mse)

    # print(pd.Series(my_model.predict(testing_data_total)))


    # my_result = pd.Series(my_model.predict(testing_data_total))
    my_result = my_model.predict(testing_data_total)

    for t in range(len(my_result)):
        if my_result[t][0] < 0 :
            result.값[a[t]] = 0
        else:
            result.값[a[t]] = my_result[t][0]
        print('*******************************************')
        print(mse)
        print('*******************************************')
        (result.metric_acc)[a[t]] = 1.0/float(mse)
    print(result)
    result.to_csv('DL_result_mat.csv', encoding = 'cp949', index=False)
    K.clear_session()
