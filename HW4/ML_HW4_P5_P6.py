#Problem 5

import numpy as np

vocab = {}
vocab_size = 0
reviews = []

with open('reviews_limited_vocab.txt', 'r') as f:
    for line in f.readlines():
        words = line.strip().split(' ')
        for word in words:
            if word not in vocab:
                vocab[word] = vocab_size
                vocab_size += 1
        reviews.append([vocab[word] for word in words])

invert_vocab = [''] * vocab_size
for (word, word_id) in vocab.items():
    invert_vocab[word_id] = word
invert_vocab = np.array(invert_vocab)

words_to_compare = ['excellent', 'amazing', 'delicious', 'fantastic', 
                    'gem', 'perfectly', 'incredible', 'worst', 
                    'mediocre', 'bland', 'meh', 'awful', 'horrible', 
                    'terrible']

k_to_try = [ 2, 4, 8 ]
print(len(reviews))

from numpy import zeros

def bag_of_words_rep(word_ids, dim):
    bow_vector = zeros(dim) # creates a numpy.ndarray of shape (dim,)
    for word_id in word_ids:
        bow_vector[word_id] += 1
    return bow_vector 

#print(reviews[0])
first_bow_vector = bag_of_words_rep(reviews[0], vocab_size)
#print(first_bow_vector)

#construct document term matrix with each row representing a document 
#and each column representing a frequency of a given word
def constructDocMat(revs):
    doc_mat=[]
    for review in revs:
        doc_mat.append(bag_of_words_rep(review, vocab_size))
    return doc_mat
doc_mat=constructDocMat(reviews)
#print(doc_mat[0])

u, s, v = np.linalg.svd(doc_mat, full_matrices=False)

from numpy.linalg import norm
def cos_sim(a,b):
    cos_sim0 = np.dot(a,b)/(norm(a, axis=0)*norm(b))
    return cos_sim0

print("In this chart the row and columns are labeled by the indeces of the words in the 'words_to_compare'")
print("The mapping goes like:")
for word in words_to_compare:
    print("\t",word,"=",words_to_compare.index(word))
#print(len(u2[0]))
def do_cossim_stuff(u0,k0):
    
    #print("k is",k0)
    u2 = u0[:,:k0]
    cos_sims=[]
    r=2
    #print(len(u2[0]))
    for word1 in words_to_compare:
        word1_index=vocab[word1]
        cos_sims_row=[]
        #print(u2[word1_index])
        for word2 in words_to_compare:
            word2_index=vocab[word2]
            cos_sims_row.append(round(cos_sim(u2[word1_index],u2[word2_index]),10))
        cos_sims.append(cos_sims_row)

    #print(cos_sims)
    str_cos_sims=[]
    
    nums=[]
    for i in range(0,14):
        if i<10:
            nums.append("  "+str(i)+"  ")
        else:
            nums.append("  "+str(i)+" ")
    num0=0
    for row in cos_sims:
        string_row=[]
        for num in row:
            if num > 0:
                string_row.append('{:.2f}'.format(round(num, r))+" ")
            else:
                string_row.append('{:.2f}'.format(round(num, r))+"")
        if num0<10:
            string_row.insert(0,str(num0)+"    ")
        else:
            string_row.insert(0,str(num0)+"   ")
        num0+=1
        str_cos_sims.append(string_row)
        
    print('     ',' '.join(nums),'\n')
    for row in str_cos_sims:
        print('|'.join(row))
    return

for ks in k_to_try:
    print("\n\n"+"k =",ks)
    do_cossim_stuff(v.T,ks)
    print("\n\n")

#PROBLEM 6 
print("In this chart the row and columns are labeled by the indeces of the words in the 'words_to_compare'")
print("The mapping goes like:")
for word in words_to_compare:
    print("\t",word,"=",words_to_compare.index(word))

up=list(v)
up.pop(0)
print("PROBLEM 6\n\n"+"the outputs after removing the first singular vector are:")
up=np.array(up)
for ks in k_to_try:
    print("\n\n"+"k =",ks)
    print("\n")
    do_cossim_stuff(up.T,ks)
    print("\n\n")
    