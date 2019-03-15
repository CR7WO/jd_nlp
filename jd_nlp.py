import pandas as pd
import jieba
from collections import Counter
import csv
import codecs
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot  as plt
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.layers import Embedding
from keras.layers import Conv1D,GlobalMaxPooling1D


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
    print(seg_text[1:3])
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
    print(sh_con_t[1:3])
    return sh_con_t

def write_csv(file_path,datas):
    f = codecs.open(file_path,'a','utf-8')
    writer = csv.writer(f)
    writer.writerows(datas)



if __name__ == '__main__':
    hp = pd.read_csv('jd_cn_hp.csv', header=None)
    cp = pd.read_csv('jd_cn_cp.csv', header=None)
    hp.columns = ['id', 'id_m', 'time', 'comment']
    cp.columns = ['id', 'id_m', 'time', 'comment']
    h_con = hp['comment']
    c_con = cp['comment']
    text = pd.concat([h_con, c_con])
    c_text = concat_text(text)
    s_text = seg_depart(c_text)
    p_text = Counter(s_text).most_common(2500)
    d_text = {key: value for (key, value) in p_text}
    sh_con = seg_depart1(h_con)
    sc_con = seg_depart1(c_con)
    h_con_bm = transform_bm(sh_con)
    c_con_bm = transform_bm(sc_con)
    #write_csv('bm_cp',c_con_bm)

    max_features = 5000
    maxlen = 100
    batch_size = 64
    embedding_dims = 50
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 20

    text_bm = []
    text_bm.extend(h_con_bm[:1100])
    text_bm.extend(c_con_bm)
    hv_l = []
    cv_l = []
    for i in range(len(h_con_bm[:1100])):
        hv_l.append(1)

    for i in range(len(c_con_bm)):
        cv_l.append(0)

    print(len(hv_l), len(cv_l))
    hcv = []
    hcv.extend(hv_l)
    hcv.extend(cv_l)
    x_train, x_test, y_train, y_test = train_test_split(text_bm, hcv, test_size=0.3)
    x_train = sequence.pad_sequences(x_train, maxlen=100)
    x_test = sequence.pad_sequences(x_test, maxlen=100)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    model = Sequential()
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
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

    history_dict = history.history
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    ep = range(1, len(acc) + 1)
    plt.plot(ep, loss, 'bo', label="Trainning loss")
    plt.plot(ep, acc, 'ro', label="Training acc")
    plt.plot(ep, val_loss, 'b', label='val loss')
    plt.plot(ep, val_acc, 'r', label='val acc')
    plt.title('Loss and Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss and Acc')
    plt.legend()
    plt.show()