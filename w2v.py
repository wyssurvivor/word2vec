import numpy as np
import collections


vocabulary_size=10000
embedding_size=150
window = 2
negative_sampling=1
cbow=True

embedding = np.random([embedding_size, vocabulary_size])
weights = np.random([vocabulary_size, embedding_size])

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
    list = range(target_index-window,target_index+window+1,1)
    input_matrix=embedding[:list]
    return np.sum(input_matrix, axis=1)/len(list)


def get_softmax(hidden_vector):
