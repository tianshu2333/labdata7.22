import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#additional module: perplexity calculation
# def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
#     """calculate the perplexity of a lda-model"""
#     # dictionary : {7822:'deferment', 1841:'circuitry',19202:'fabianism'...]
#     print ('the info of this ldamodel: \n')
#     print ('num of testset: %s; size_dictionary: %s; num of topics: %s'%(len(testset), size_dictionary, num_topics))
#     prep = 0.0
#     prob_doc_sum = 0.0
#     topic_word_list = [] # store the probablity of topic-word:[(u'business', 0.010020942661849608),(u'family', 0.0088027946271537413)...]
#     for topic_id in range(num_topics):
#         topic_word = ldamodel.show_topic(topic_id, size_dictionary)
#         dic = {}
#         for word, probability in topic_word:
#             dic[word] = probability
#         topic_word_list.append(dic)
#     doc_topics_ist = [] #store the doc-topic tuples:[(0, 0.0006211180124223594),(1, 0.0006211180124223594),...]
#     for doc in testset:
#         doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
#     testset_word_num = 0
#     for i in range(len(testset)):
#         prob_doc = 0.0 # the probablity of the doc
#         doc = testset[i]
#         doc_word_num = 0 # the num of words in the doc
#         for word_id, num in doc.items():
#             prob_word = 0.0 # the probablity of the word
#             doc_word_num += num
#             word = dictionary[word_id]
#             for topic_id in range(num_topics):
#                 # cal p(w) : p(w) = sumz(p(z)*p(w|z))
#                 prob_topic = doc_topics_ist[i][topic_id][1]
#                 prob_topic_word = topic_word_list[topic_id][word]
#                 prob_word += prob_topic*prob_topic_word
#             prob_doc += math.log(prob_word) # p(d) = sum(log(p(w)))
#         prob_doc_sum += prob_doc
#         testset_word_num += doc_word_num
#     prep = math.exp(-prob_doc_sum/testset_word_num) # perplexity = exp(-sum(p(d)/sum(Nd))
#     print ("the perplexity of this ldamodel is : %s"%prep)
#     return prep


data = pd.read_csv('C:/Users/lhy12/Desktop/PCA/merge/HA_HpA_merge_T.csv')

Feature_data = np.array(data.iloc[:, 1:1081])
Target_data = np.array(data.iloc[0:45, 1])
scaler = preprocessing.StandardScaler().fit(Feature_data)
#标准化数据
Feature_data_standard = scaler.transform(Feature_data)
#print(Feature_data_standard)
print(Target_data)
print(type(Feature_data_standard))
print(type(Feature_data))

#accuracy setting
h = .01
clf = LinearDiscriminantAnalysis(n_components=2)


# x_min, x_max = Feature_data_standard[, 0].min() - 1, Feature_data_standard[, 0].max() + 1
# y_min, y_max = Feature_data_standard [, 1].min() - 1, Feature_data_standard[, 1].max() + 1


clf.fit(Feature_data_standard, Target_data)
Feature_data_standard = clf.transform(Feature_data_standard)
a = plt.scatter(Feature_data_standard[0:9, 0], Feature_data_standard[0:9, 1], marker='o', c='blue' )
b = plt.scatter(Feature_data_standard[9:18, 0], Feature_data_standard[9:18, 1], marker='o', c='yellow' )
c = plt.scatter(Feature_data_standard[18:27, 0], Feature_data_standard[18:27, 1], marker='o', c='black' )
d = plt.scatter(Feature_data_standard[27:36, 0], Feature_data_standard[27:36, 1], marker='o', c='red' )
e = plt.scatter(Feature_data_standard[36:45, 0], Feature_data_standard[36:45, 1], marker='o', c='orange' )
plt.legend((a, b, c, d, e), (u'HA_HpA', u'HA', u'HA_OA', u'HpA', u'HpA_OA'))
plt.suptitle("LDA Dimensionality Reduction")
plt.title("Digit Dataset")
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
# plt.figure(figsize=(60,40))
# plt.show()
plt.savefig('fig_dog.png')
plt.show()