import pandas as pd
import numpy as np
import jieba
from snownlp import SnowNLP
from wordcloud import WordCloud
import random
from collections import Counter
import csv
import codecs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot  as plt
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.layers import Embedding,LSTM
from keras.layers import Conv1D,MaxPooling1D

##停用词表
def stopwordslist():
    stopwords = [line.strip() for line in open('哈工大停用词表.txt',encoding='UTF-8')
                .readlines()]
    return stopwords

##拼接字符串
def concat_text(text):
    c_text = ''
    for line in text:
        c_text = c_text + ',' + line
    return c_text

##字符串分词（频数）
def seg_depart(text):
    seg_text=[]
    text_ja = jieba.lcut(text)
    stopwords = stopwordslist()
    for word in text_ja:
        if word not in stopwords:
            seg_text.append(word)
    return seg_text

##评论分词
def seg_depart1(text):
    seg_text=[]
    stopwords = stopwordslist()
    for li in text:
        lt = []
        text_ja = jieba.lcut(li)
        for word in text_ja:
            if word not in stopwords:
                lt.append(word)
        seg_text.append(lt)
    #print(seg_text[1:3])
    return seg_text

##频数转编码
def get_bm(d_text):
    l_key = []
    l_value = []
    for key in d_text.keys():
        l_key.append(key)
        l_value.append(d_text.get(key))

    bm = [i + 1 for i in range(2500)]
    kv = {'word': l_key, 'count': l_value, 'bm': bm}
    c_word = pd.DataFrame(kv)
    return c_word

##词转编码
def transform_bm(sh_text):
    sh_con_t = []
    c_word = get_bm(d_text)
    for line in sh_text:
        con = []
        for word in line:

            if word in list(c_word['word']):
                num = int(c_word[c_word['word'] == word]['bm'])
                con.append(num)
            else:
                con.append(0)
        sh_con_t.append(con)
    #print(sh_con_t[1:3])
    return sh_con_t

def write_csv(file_path,datas):
    f = codecs.open(file_path,'a','utf-8')
    writer = csv.writer(f)
    writer.writerows(datas)


if __name__ == '__main__':
    ##处理数据
    hp = pd.read_csv('jd_cn_hp.csv', header=None)
    cp = pd.read_csv('jd_cn_cp.csv', header=None)
    # pre_text = pd.read_csv('shop_c.csv')
    hp.columns = ['id', 'id_m', 'time', 'comment']
    cp.columns = ['id', 'id_m', 'time', 'comment']
    hp['y_label'] = 1
    cp['y_label'] = 0
    hp_t = hp.iloc[:1100, 3:]
    cp_t = cp.iloc[:, 3:]
    print(hp_t.shape, cp_t.shape)
    print(hp_t.head(), cp_t.head())
    hc = pd.concat([hp_t, cp_t])
    print(hc.shape)
    text = hc['comment']
    c_text = concat_text(text)
    s_text = seg_depart(c_text)
    p_text = Counter(s_text).most_common(2500)
    d_text = {key: value for (key, value) in p_text}
    ##查看频数最多的前20个关键词
    i = 1
    for k, v in d_text.items():
        if i <= 20:
            print(k, ':', v)
        i += 1
    ##划分训练集&测试集&验证集
    xl = hc['comment']
    yl = hc['y_label']
    x_train_l, x_test_l, y_train, y_test_l = train_test_split(xl, yl, test_size=0.3)
    x_test_t, x_pre, y_test, y_label = train_test_split(x_test_l, y_test_l, test_size=0.2)

    xr_con = seg_depart1(x_train_l)
    xe_con = seg_depart1(x_test_t)
    pre_con = seg_depart1(x_pre)
    print(xr_con[:5])
    print(xe_con[:5])
    print(pre_con[:5])

    xr_con_bm = transform_bm(xr_con)
    xe_con_bm = transform_bm(xe_con)
    pre_con_bm = transform_bm(pre_con)
    print(xr_con_bm[:5])
    print(xe_con_bm[:5])
    print(pre_con_bm[:5])


    max_features = 5000
    maxlen = 100
    batch_size = 32
    embedding_dims = 30
    #filters = 64
    #kernel_size = 3
    hidden_dims = 250
    epochs = 10

    #text_bm = []
    #text_bm.extend(h_con_bm[:1100])
    #text_bm.extend(c_con_bm)
    #hv_l = []
    #cv_l = []
    #for i in range(len(h_con_bm[:1100])):
    #    hv_l.append(1)
    #for i in range(len(c_con_bm)):
    #    cv_l.append(0)
    #print(len(hv_l), len(cv_l))
    #hcv = []
    #hcv.extend(hv_l)
    #hcv.extend(cv_l)
    #x_train, x_test1, y_train, y_test1 = train_test_split(text_bm, hcv, test_size=0.3)
    #x_test, pre_x, y_test, y_lab = train_test_split(x_test1,y_test1,test_size=0.2)
    x_train = sequence.pad_sequences(xr_con_bm, maxlen=maxlen)
    x_test = sequence.pad_sequences(xe_con_bm, maxlen=maxlen)
    pre_x = sequence.pad_sequences(pre_con_bm,maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('pre_x shape:', pre_x.shape)


    ##建立模型
    model = Sequential()
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(Conv1D(72,
                     5,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(Conv1D(72,
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D())
    model.add(LSTM(100))
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    ##画图-误差与准确率
    history_dict = history.history
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig1 = plt.figure(num='fig1',figsize=(10,8))
    fig2 = plt.figure(num='fig2', figsize=(10, 8))
    fig3 = plt.figure(num='fig3', figsize=(10, 8))

    plt.figure(num='fig1')
    ep = range(1, len(acc) + 1)
    plt.plot(ep, loss, 'bo', label="Trainning loss")
    plt.plot(ep, acc, 'ro', label="Training acc")
    plt.plot(ep, val_loss, 'b', label='val loss')
    plt.plot(ep, val_acc, 'r', label='val acc')
    plt.title('Loss and Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss and Acc')
    plt.legend()
    #plt.show()

    pre_y = model.predict(pre_x, batch_size=32)
    pre_y_class = model.predict_classes(pre_x, batch_size=32)
    pre_y = [round(x[0], 3) for x in pre_y]
    pre_y_class = [x[0] for x in pre_y_class]
    print(pre_y[:10])
    print(pre_y_class[:10])

    # 关键词情感分类
    star = []
    for it in x_pre:
        s = SnowNLP(it)
        t = s.sentiments
        star.append(round(t, 3))
    print(star[:10])

    ##汇总
    star_class = np.array(star)
    star_class = np.where(star_class >= 0.5, 1, 0).tolist()
    x_pre = x_pre.tolist()
    y_label = y_label.tolist()
    # pre_y = pre_y.tolist()
    # pre_y_class = pre_y_class.tolist()

    # 验证模型准确率
    acc1 = accuracy_score(y_label, pre_y_class)
    acc2 = accuracy_score(y_label,star_class)
    print("自建深度模型准确率：{}".format(round(acc1,3)))
    print("snownlp模型准确率：{}".format(round(acc2, 3)))
    print(type(star_class),
          type(x_pre),
          type(y_label),
          type(pre_y),
          type(pre_y_class),
          type(star))
    dicts = {'comment': x_pre, 'y_label': y_label, 'predict': pre_y,
             'class': pre_y_class, 'star': star, 'star_class': star_class}
    df_pre = pd.DataFrame(dicts)
    print(df_pre.head())

    df_l = df_pre.loc[df_pre['class'] != df_pre['star_class'], ['comment', 'y_label', 'class', 'star_class']]
    print(df_l)
    df_l.to_csv('no_class.csv',index=False)
    ##绘制词云
    hp_t_c = hp_t['comment']
    cp_t_c = cp_t['comment']
    hp_s = " ".join(seg_depart(concat_text(hp_t_c)))
    cp_s = " ".join(seg_depart(concat_text(cp_t_c)))
    ##好评
    wc1 = WordCloud(mask=plt.imread('3.jpg'),
                   background_color='white',
                   )
    plt.figure(num='fig2')

    plt.axis('off')
    plt.imshow(wc1.generate(hp_s))
    #plt.show()

    ##差评
    wc2 = WordCloud(mask=plt.imread('3.jpg'))
    plt.figure(num='fig3')
    plt.axis('off')
    plt.imshow(wc2.generate(cp_s))
    plt.show()