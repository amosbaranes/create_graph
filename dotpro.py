import io
import codecs
import numpy as np
from sklearn.preprocessing import normalize
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mla
import time
from scipy.stats import norm
import seaborn as sns
from scipy.stats import norm
import csv

# for cosine similarity
from scipy import spatial
from math import *
import pandas as pd
from Threshold import get_connection


def load_vectors_array_imp(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    s = fin.readline().split()
    print(s)
    if len(s) == 2:
        n, d = map(int, s)
    else:
        print("error")
        n, d = map(int, s)
    print(n, d)
    word2id = {}
    words = list()
    A = np.zeros((n, d), dtype=float)
    # B = np.zeros((n,d), dtype=float)
    print(A.shape)
    i = 0
    for line in fin:
        tokens = line.rstrip().split()
        # data[tokens[0]] = map(float, tokens[1:])
        word = tokens[0]
        embedding = np.array([float(val) for val in tokens[1:]])
        assert (embedding.size == d)
        words.append(word)
        A[i] = embedding
        # normalized = embedding / np.linalg.norm(embedding)
        # B[i] = normalized
        # print(word)
        # print(i)
        word2id[word.strip()] = i
        i = i+1
#        A = np.append(A, embedding)
#        A.append(embedding)
#        data[tokens[0]] = embedding
    assert(i == n)
    # B = normalize(A, norm='l2')

    # print (A.shape)
    # print (A[0])
    # print (words[0])
#    print (data.())
#    assert(len(data)==n)
    # verify n,d
    return (A, words, word2id)


def cosine_similarity(w1, w2):
    return np.dot(w1, w2)/(np.linalg.norm(w1) * np.linalg.norm(w2))


def get_data(path):
    # nrow means how many rows from the txt model you take
    data = (pd.read_csv(path, sep=" "))
    words = data["say_VERB"]
    del(data["say_VERB"])
    return np.array(data), words


if __name__ == '__main__':

    # (A, words) = get_data("model.txt")
    (A, words, word2id) = load_vectors_array_imp("data/model.2.w.txt")
    # print(np.array(words))
    # print(np.array(A)) # the vectors
    # print(np.array(word2id)) # the words dictionary
    all_words = {}
    good_words = {}
    k = 0
    print(np.array(list(word2id.keys())))
    w = np.array(list(word2id.keys()))
    print(w[0])
    print(len(w))

    count_of_words = 9000
    d = get_connection(A[:count_of_words], words, A)
    # d = get_connection(A, words, A)
    # for key in d.keys():
    #     print("word: " + str(key) + ", words: " + str(d[key]))
    """"
    print("start")
    (A, words) = get_data("model.txt")
    count_of_words = 100
    d = get_connection(A[:count_of_words], words, A)
    for key in d.keys():
        print("word: " + str(key) + ", words: " + str(d[key]))


    # calculate the similarity between all the words in the data
    for word in words:
        for i in range(len(words)):
            if i != k:
                s = cosine_similarity(A[k], A[i])
                if s > 0:
                    good_words[words[i]] = '{:.5f}'.format(s)

        k += 1

        all_words[word] = good_words
        good_words = {}

    for key in all_words.keys():
        # sort the words and keep the 15 highest values
        temp = {k: v for k, v in sorted(
            all_words[key].items(), key=lambda item: item[1], reverse=True)}
        all_words[key] = temp

        if len(all_words[key]) > 15:
            temp = {k: v for k, v in sorted(
                all_words[key].items(), key=lambda item: item[1], reverse=True)}
            all_words[key] = {}

            count = 0
            for key1 in temp.keys():
                count += 1
                all_words[key][key1] = temp[key1]
                if count >= 15:
                    break
        # print the word and the similarity words
        print("word: " + str(key) +
              ", similarity words: " + str(all_words[key]))


"""
"""
    m = np.mean(A, axis=0)
    print("mean:\n")
    print(m)
    if False:  # demean?
        B = A - m
    else:
        B = A
    # B = demean(A, axis=0)
    print("mean demeaned:\n")
    print(np.mean(B, axis=0))

#    print (B[i])
    C = normalize(B, norm='l2', axis=0)
    # print (C[i])
    print("mean normalized:\n")
    print(np.mean(C, axis=0))

    nstd = 3
    n = C.shape[0]

    node_stats = {}
    with open("word_stats.csv", 'w', newline='', encoding='utf-8') as csvfile:
        print("writing word stats")
        writer = csv.writer(csvfile)
        writer.writerow(["index", "word", "min", "mean", "max", "std"])

        print("size:", n)
        for i in range(n):
            dp = np.dot(C, C[i])
            distances = [dp[k] for k in range(n) if k != i]
            (mean, std) = (np.mean(distances), np.std(distances))
            (mn, mx) = (min(distances), max(distances))
            writer.writerow([i, words[i], mn, mean, mx, std])
            node_stats[i] = mean+nstd*std

#            edges = [words[k] for k in range(n) if k!=i and dp[k]>h["mean"]+3* h["std"]]
#            edges2 = [words[k] for k in range(n) if k!=i and dp[k]<h["mean"]-3* h["std"]]
#            print (i, words[i], len(edges), edges, edges2)

    with open("graph.csv", 'w', newline='', encoding='utf-8') as csvfile, open("graph_w.csv", 'w', newline='', encoding='utf-8') as csvfile_w:
        writer = csv.writer(csvfile)
        writer_w = csv.writer(csvfile_w)

        for i in range(n):
            dp = np.dot(C, C[i])
            edges = [k for k in range(
                n) if k != i and dp[k] > node_stats[i] and dp[k] > node_stats[k]]
            row = [i, len(edges)]
            row.append(edges)
            writer.writerow(row)

            if (len(edges) > 0):
                edges = [words[k] for k in range(
                    n) if k != i and dp[k] > node_stats[i] and dp[k] > node_stats[k]]
                row = [words[i], len(edges)]
                row.append(edges)
                writer_w.writerow(row)


##
    # h = {}
    # (h["mean"], h["std"], h["min"], h["max"], h["index"], h["word"], h["nstd"]) = (None, None, None, None, None, None, None)
    # with open("word_stats.csv", 'w', newline='', encoding='utf-8') as csvfile:
    #     print ("writing word stats")
    #     writer = csv.DictWriter(csvfile, h.keys())
    #     writer.writeheader()
    #
    #     print ("size:", n)
    #     for i in range(n):
    #         dp = np.dot(C, C[i])
    #         distances = [dp[k] for k in range(n) if k!=i]
    #         (h["mean"], h["std"]) = (np.mean(distances), np.std(distances))
    #         (h["min"], h["max"]) = (min(distances), max(distances))
    #         (h["index"], h["word"], h["nstd"]) = (i, words[i], nstd* h["std"])
    #         writer.writerow(h)
    #         node_stats[i] = h["mean"]+nstd*h["std"]
    #         h = {}
"""
