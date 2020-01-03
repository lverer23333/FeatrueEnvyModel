
import numpy as np
import time
import os
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json  

MAX_SEQUENCE_LENGTH = 15 

"""
    本项目中涉及四类文件：
    主目录文件：
        test_ClssId：记录所有id对应的测试用例的标签值
        targetClasses：记录每一个为正（有坏味）的用例中，方法所对应的正确的目标类
    针对每个测试用例（方法）：包括该方法能移动到的所有类，每一行都表示一个目标类的参数
        test_Distances：每一行表示该方法到该类的距离，该方法到目标类的距离，该用例的标签。注意此处的标签和test_ClssId中的标签是可以对应上的
        test_Names：每一行表示该方法名字，该方法所在类名字，目标类名字
"""

# 用来测试的项目
TESTPATH = '../Data/1#Fold/test-junit/'
# 用来测试的模型
MODELPATH = ''
FILENAME = '../Data/1#Fold/test-junit/test_ClssId.txt'
TARGETPATH = '../Data/1#Fold/test-junit/targetClasses.txt'
values = []
predsTargetClassNames = []
print ("start time:"+time.strftime("%Y/%m/%d  %H:%M:%S"))
start = time.clock()

'''神经网络所需的传入信息：①当前方法的名字、当前方法所在类的名字，目标类的名字 ②当前方法到当前类的距离，当前方法到目标类的距离'''

# 记录每一个为正（有坏味）的用例中，方法所对应的正确的目标类
f = open(TARGETPATH, 'r', encoding = 'utf-8')
for line in f:
    predsTargetClassName = line.split()
    predsTargetClassNames.append(predsTargetClassName)

# 记录所有id对应的测试用例的标签值
f = open(FILENAME, 'r', encoding = 'utf-8')
for line in f:
    value = line.split()
    values.append(value)

TP = 0 
FN = 0 
FP = 0 
TN = 0 
NUM_CORRECT = 0
TOTAL = 0
model = model_from_json(open(MODELPATH + 'my_model.json').read())
model.load_weights(MODELPATH + 'my_model_weights.h5')
ii = 0

for sentence in values:
    ii=ii+1
    test_distances = []
    test_labels = []
    test_texts = []
    targetClassNames=[]
    classId = sentence[0]
    label = sentence[1]

    if(os.path.exists(TESTPATH + 'test_Distances'+classId+'.txt')):
        # 针对每个用例，每一行表示该方法到该类的距离，该方法到目标类的距离，该用例的标签。注意此处的标签和test_ClssId中的标签是可以对应上的
        with open(TESTPATH + 'test_Distances'+classId+'.txt','r') as file_to_read:
            for line in file_to_read.readlines():
                values = line.split()
                test_distance = values[:2]
                test_distances.append(test_distance)
                test_label =values[2:]
                test_labels.append(test_label)

        # 针对每个用例，每一行表示该方法名字，该方法所在类名字，目标类名字
        with open(TESTPATH + 'test_Names'+classId+'.txt','r') as file_to_read:
            for line in file_to_read.readlines():
                test_texts.append(line)
                line = line.split()
                targetClassNames.append(line[10:])

        # 使用Tokenizer建立基于词典位序的文本向量表示
        tokenizer1 = Tokenizer(num_words=None)
        tokenizer1.fit_on_texts(test_texts)
        test_sequences = tokenizer1.texts_to_sequences(test_texts)
        test_word_index = tokenizer1.word_index
        test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

        # 将结构数据转成asarray
        test_distances = np.asarray(test_distances)
        test_labels = np.asarray(test_labels)

        # 定义train set
        x_val = []
        x_val_names = test_data
        x_val_dis = test_distances
        x_val_dis = np.expand_dims(x_val_dis, axis=2)
        x_val.append(x_val_names)
        x_val.append(np.array(x_val_dis))
        y_val = np.array(test_labels)

        # 进行预测
        preds = model.predict_classes(x_val) # 获取preds的值（1或0）
        preds_double = model.predict(x_val) # 获取精度为double的preds的值（为1的概率，为0的概率）
        NUM_ZERO = 0
        NUM_ONE = 0
        MAX = 0
        for i in range(len(preds)):
            # preds[i]是第i个例子的预测结果，只要有一个预测出有坏味（NUM_ONE！=0），就是有坏味；全都是无坏味（NUM_ONE=0），才是无坏味
            if(preds[i]==0):
                NUM_ZERO += 1   # 预测结果是负，无坏味
            else:
                NUM_ONE += 1    # 预测结果是正，有坏味
        if(len(preds)!=0 and label == '1'): # 该用例是正（有坏味），判断进行到predsTargetClassNames中的第几个用例了
            TOTAL+=1
        if(label == '1' and NUM_ONE == 0): # 该用例是正但预测为负了（即本来有坏味预测成无坏味）
            FN += 1
        if(label == '1' and NUM_ONE != 0): # 该用例是正且预测为正（即成功预测有坏味）
            TP+=1;
            # 看是不是正确预测到需要移动的类了
            correctTargets = []
            # preds_double[i][0]表示第i个用例为1的概率是多少，选出为一的概率中最大的那个类，将其附加到correctTargets，然后逐行对比当前列表是否和给出的predsTargetClassNames相同
            for i in range(len(preds_double)):
                if(preds_double[i][0]>=MAX):
                    MAX = preds_double[i][0]
            for i in range(len(preds_double)):
                if(preds_double[i][0] == MAX):
                    correctTargets.append(targetClassNames[i])
            for i in range(len(correctTargets)):
                # Total无论预测结果是否准确都会+1
                if(correctTargets[i]==predsTargetClassNames[TOTAL-1]):
                    NUM_CORRECT += 1
                    break
        if(label == '0' and NUM_ONE == 0): # 该用例是负且预测为负（即成功预测没有坏味）
            TN += 1
        if(label == '0' and NUM_ONE !=0): # 该用例是负且预测为正（即本来没有坏味预测成有坏味）
            FP += 1

print('TP--------', TP)
print('TN--------', TN)
print('FP--------', FP)
print('FN--------', FN)
print('NUM_ZERO---', NUM_ZERO)
print('NUM_ONE---', NUM_ONE)
print('NUM_CORRECT----',NUM_CORRECT)
print('TargetAccuracy---',NUM_CORRECT/TP)
if(TP+FP!=0):
    print('Test Precision',TP/(TP+FP))
else:
    print('Test Precision',0)
if(TP+FN!=0):
    print('Test Recall',TP/(TP+FN))
else:
    print('Test Recall',0)
end = time.clock()

print('Running time: %s Seconds'%(end-start))


print ("end time:"+time.strftime("%Y/%m/%d  %H:%M:%S"))