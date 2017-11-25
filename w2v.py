#coding:utf8
import numpy as np
import collections
import random


vocabulary_size=10000
embedding_size=150
window = 2
negative_sampling=5
cbow=True

learning_rate=0.01

embedding = np.random.rand(embedding_size, vocabulary_size)
weights = np.random.rand(vocabulary_size, embedding_size)

def get_data(words):
    count=[['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))

    dictionary={}
    for word,_ in count:
        dictionary[word]=len(dictionary)

    data = list()
    unk_count=0
    for word in words:
        if word in dictionary:
            index=dictionary[word]
        else:
            unk_count+=1
            index=0
        data.append(index)

    count[0][1]=unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return count, data, dictionary, reverse_dictionary

words=[] #todo
count,data,dictionary,reverse_dictionay=get_data(words)


#input-》hidden
def get_hidden(target_index):
    list = range(max(target_index-window,0),min(target_index+window+1, len(data)),1)
    del list[window] # 删掉w对应的坐标
    word_index_list=data[list]
    input_matrix=embedding[:,word_index_list] #根据下标拿出没个context w对应的向量
    return np.sum(input_matrix, axis=1)/len(list) # 没个context w对应的向量相加生成hidden layer的输入

#hidden->y
def get_softmax(hidden_vector):
    return

# get nums negative sampling
def get_negative_samples(nums, skip_index, high, low=0):
    count = 0
    negative_index=[]
    negative_index.append(skip_index)
    while count<nums:
        index=random.randint(low, high)
        if index in negative_index:
            continue
        negative_index.append(index)
        count+=1
    del negative_index[0]
    return negative_index

def sigmoid_single(x):
    return 1.0/(1+np.exp(-x))

def update_context_word_vectors(e, target_index):
    list = range(max(target_index - window, 0), min(target_index + window + 1, len(data)), 1)
    del list[window]  # 删掉w对应的坐标
    for index in list:
        word_index=data[index]
        embedding[word_index]=np.add(embedding[word_index], e)

def train():

    for index,word_index in enumerate(data):
        target_list=get_negative_samples(negative_sampling, word_index, len(dictionary.keys())-1)
        target_list.append(word_index)
        x=get_hidden(index)
        e=np.zeros([embedding_size,1])
        labels=np.zeros([2*window,1])
        labels=np.append(labels, 1)
        for u_index,u in enumerate(target_list):
            theta_u = weights[u]
            q_target=sigmoid_single(np.dot(x.T, theta_u))
            g=learning_rate*(labels[u_index]-q_target)
            e=np.add(e, g*theta_u)
            weights[u]=np.add(theta_u, g*x)

        update_context_word_vectors(e, index)





if __name__=='__main__':
    train()