import pandas as pd
import nltk
import re

filename = "question1_sentiment.csv"

document_info = pd.read_csv(filename)
# print(document_info.head())

content = document_info['review'].tolist()
# print(content[0])
# print(content[0:10])

content_S = []
# for line in content:
#    current_segment = nltk.word_tokenize(line)
#    if len(current_segment) >= 1:
#        content_S.append(current_segment)

# 去除html标签，'\', 英文缩写, 非英文字符
current_segment = nltk.word_tokenize(re.sub(r"</?[^>]*>|\\|'\w*|[^(\w|\s)]", ' ', content[2]))
content_S.append(current_segment)

print(content[2])
print(content_S[0])
