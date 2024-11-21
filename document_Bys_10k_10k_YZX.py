import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 数据文件
filename = "question1_sentiment.csv"

# 输出目录、文件
output_final = "BOW_predict_Bys.txt"
output_orig = "IMDB_test_label.txt"

# 创建文件并存储list对象
contentFile_All = 'contentFile_All.dat'
f = open(contentFile_All, 'rb')
content_all = pickle.load(f)

# print(content_all)
# print('长度：', len(content_all))
# print(content_all[0])
# print(len(content_all[0]))

document_info = pd.read_csv(filename)
label = document_info['label'].values.tolist()
review = content_all

df_train = pd.DataFrame({'review': review, 'label': label})
# print(df_train.head())
# print(df_train.loc[19999])
# print(df_train.loc[20000])

# print('(********************************)')

# 模型选择
x_train, x_test, y_train, y_test = train_test_split(df_train['review'].values,
                                                    df_train['label'].values, train_size=10000,
                                                    test_size=10000, shuffle=False, random_state=1)

# 处理训练集特征
words = []
for line_index in range(len(x_train)):
    try:
        words.append(' '.join(x_train[line_index]))
    except:
        print(line_index, x_train[line_index])
# print(words[0])

# 特征提取
#
vectorizer = CountVectorizer(analyzer='word', max_features=1000, lowercase=False)
# vectorizer = CountVectorizer(analyzer='word', max_features=4000, lowercase= False)
# vectorizer = TfidfVectorizer(analyzer='word', max_features=4000, lowercase= False)
# vectorizer = TfidfVectorizer(analyzer='word', max_features=1000, lowercase= False)
vectorizer.fit(words)

# 生成分类器
classifier = MultinomialNB()
classifier.fit(vectorizer.transform(words), y_train)
# 处理测试集特征
test_words = []
for line_index in range(len(x_test)):
    try:
        test_words.append(' '.join(x_test[line_index]))
    except:
        print(line_index, x_test[line_index])

# 预测
predict_result = classifier.predict(vectorizer.transform(test_words))
# 正确率
accuracy = metrics.accuracy_score(predict_result, y_test)
print("预测的准确率为:%f" % accuracy)

# 查准率，查全率，f1_score
prec_rate = metrics.precision_score(predict_result, y_test)
rec_rate = metrics.recall_score(predict_result, y_test)
f1_score = metrics.f1_score(predict_result, y_test)
print("precision_rate:%.4f" % prec_rate)
print("recall_rate:%.4f" % rec_rate)
print("f1_score:%.4f" % f1_score)

print('测试集上的预测结果：')
print(predict_result)
orig_result = y_test
# classifier.predict(vectorizer.transform(test_words))[0:10000]
predict_prob = classifier.predict_proba(vectorizer.transform(test_words))

list_orig = orig_result.tolist()
list_final = predict_prob[:, 1].tolist()

print('values and length of list_final:')
print(list_final[0:10000])
print(len(list_final))

list_id = []
for i in range(10001, 20001, 1):
    list_id.append(i)

df_orig = pd.DataFrame({'id': list_id, 'sentiment': list_orig})
df_final = pd.DataFrame({'id': list_id, 'sentiment': list_final})
df_orig.to_csv(output_orig, index=False)
df_final.to_csv(output_final, index=False)

# 计算auc并画roc曲线
fpr, tpr, threshold = roc_curve(list_orig, list_final)
roc_auc = auc(fpr, tpr)  # 计算auc的值
print('auc=%.4f' % roc_auc)
lw = 2  # 设置线宽line width
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
# 在(0,0)和(1,1)之间画折线线段
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# 设置横坐标和纵坐标的范围
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
