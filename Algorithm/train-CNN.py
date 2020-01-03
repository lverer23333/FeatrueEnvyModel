
import numpy as np
import time

np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, Embedding
from keras.models import Sequential
from keras.layers import Merge

print("start time: " + time.strftime("%Y/%m/%d  %H:%M:%S"))
distances = []  # TrainSet
labels = []  # 0/1
texts = []  # ClassNameAndMethodName
MAX_SEQUENCE_LENGTH = 15
EMBEDDING_DIM = 200  # Dimension of word vector

'''
    处理输入信息，包括训练好的word2vec、度量量、语料信息，注意传入的信息都是已经经过处理好的用例，一共57936个
'''

print('Indexing word vectors.')
embeddings_index = {}
# 本文件为word2vec训练后的对每类词的坐标，其中一行第一个词为单词，后面的200个数字为该词的坐标
f = open('word2vecNocopy.200d.txt')
# 将f中的格式整理规范后存储到embeddings_index字典中
for line in f:
    values = line.split()   # 将每行按照空格分隔成数组
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32') #相当于创建values的副本
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# 本文件为传入CNN的数字度量，前两个为距离（本方法到本类的距离，本方法到对方类的距离），最后一个为标签，0表示负类1表示正类
with open('../Data/1#Fold/train-junit/train_Distances.txt', 'r') as file_to_read:
    for line in file_to_read.readlines():
        values = line.split()
        distance = values[:2] # 取values前2个数据
        distances.append(distance)
        label = values[2:] # 取values最后一个数据
        labels.append(label)

# 本文件为传入CNN的语料度量，并在后面使用Tokenizer进行生成向量表示
with open('../Data/1#Fold/train-junit/train_Names.txt', 'r') as file_to_read:
    for line in file_to_read.readlines():
        texts.append(line)

print('Found %s train_distances.' % len(distances))

# text.Tokenizer类用于对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示，这儿使用texts作为需要处理的语料信息
# 详细可看https://blog.csdn.net/ximibbb/article/details/79149887中的示例代码
tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(texts)   # 使用一系列文档来生成token词典，texts为list类，每个元素为一个文档
sequences = tokenizer.texts_to_sequences(texts) # 将多个文档转换为word下标的向量形式,shape为[len(texts)，len(text)] -- (文档数，每条文档的长度)
word_index = tokenizer.word_index # 一个目录索引字典，在调用fit_on_texts之后设置可将单词（字符串）映射为它们的排名或者索引
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) # 将多个序列截断或补齐为相同长度

# 将结构数据转成asarray
distances = np.asarray(distances)
labels = to_categorical(np.asarray(labels)) # 转化成二进制类矩阵
labels = labels.reshape(data.shape[0],2) # 强行reshape

print('Shape of train_data tensor:', data.shape) # Shape of train_data tensor: (57936, 15), 即用例数*Tokenizer处理后的向量表示长度
print('Shape of train_label tensor:', labels.shape) # Shape of train_label tensor: (57936, 1, 2)

# 定义train set
x_train = [] # x_train是个list，可以塞多个shape不同的
x_train_names = data # shape (57936, 15)
x_train_dis = distances # shape (57936, 2)
'''为什么要增加维度'''
x_train_dis = np.expand_dims(x_train_dis, axis=2) # 增加了一个维度，shape (57936, 2, 1)
x_train.append(x_train_names)
x_train.append(np.array(x_train_dis))
y_train = np.array(labels)

# 将项目中的单词和word2vec训练好的向量进行映射，结果存储在embedding_matrix中，shape
nb_words = len(word_index)  # 有多少个不重复单词
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))  # 不重复单词数*单个单词设定的维度,shape (2392, 200)
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:    # word2vec中已经有记录的就植入，没有的就用零向量填充
        embedding_matrix[i] = embedding_vector

# Embedding层，只能作为模型的第一层，将语料类输入和embedding_matrix的值对应上
embedding_layer = Embedding(nb_words + 1,   # input_dim：字典长度，即输入数据最大下标+1
                            EMBEDDING_DIM,  # output_dim：全连接嵌入的维度
                            input_length=MAX_SEQUENCE_LENGTH,   # input_length：当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断。
                            weights=[embedding_matrix], # 权重为之前word2vec中训练好的
                            trainable=False)

'''
    搭建训练模型
'''
print('Training model.')

model_left = Sequential()   # 代表这是一个线性模型，即多个网络层的线性堆叠
model_left.add(embedding_layer)
model_left.add(Conv1D(128, 1, padding="same", activation='tanh'))   #添加一维卷积层
model_left.add(Conv1D(128, 1, activation='tanh'))
model_left.add(Conv1D(128, 1, activation='tanh'))
model_left.add(Flatten())

model_right = Sequential()
model_right.add(Conv1D(128, 1, input_shape=(2, 1), padding="same", activation='tanh'))
model_right.add(Conv1D(128, 1, activation='tanh'))
model_right.add(Conv1D(128, 1, activation='tanh'))
model_right.add(Flatten())

merged = Merge([model_left, model_right], mode='concat')
model = Sequential()
model.add(merged)  # add merge
model.add(Dense(128, activation='tanh'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

print("start training:" + time.strftime("%Y/%m/%d  %H:%M:%S"))
model.fit(x_train, y_train, epochs=6)

score = model.evaluate(x_train, y_train, verbose=0)
print('train loss:', score[0])
print('train accuracy:', score[1])

json_string = model.to_json()
open('my_model.json', 'w').write(json_string)
model.save_weights('my_model_weights.h5')

print("end time:" + time.strftime("%Y/%m/%d  %H:%M:%S"))
