import numpy as np
import csv

"""
This code is the threshold we define for the word embedding.
we also create a csv file called "graph_idea" which has 3 coloums
first is the word, second is the Neighbors, third is the degree of the word
we also calculate the minimum, maximum and Avarage degree in the graph


"""


def cosine_similarity(w1, w2):
    return np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))


def sort_dict(d):
    sorted_dict = {}
    sorted_keys = sorted(d, key=d.get, reverse=True)
    for w in sorted_keys:
        sorted_dict[w] = d[w]
    return sorted_dict


def merge_dict(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d


def list_avg(l):
    s = 0
    for item in l:
        s += item
    return (s / len(l))


def create_dict(l1, l2, l3, l4, l5):
    d = {}
    for i in range(len(l1)):
        d[l1[i][0]] = l1[i][1]
    for i in range(len(l2)):
        d[l2[i][0]] = l2[i][1]
    for i in range(len(l3)):
        d[l3[i][0]] = l3[i][1]
    for i in range(len(l4)):
        d[l4[i][0]] = l4[i][1]
    for i in range(len(l5)):
        d[l5[i][0]] = l5[i][1]
    return d


def check_lists(l, limit):
    l = sorted(l, key=lambda x: x[1], reverse=True)
    tmp = []
    for i in range(min(limit, len(l))):
        tmp.append(l[i])
    return tmp


""" make connection between all the words that have similarity above 0.6 """


def stage_one(A, word_index, words, similarity):
    d = {}
    for i in range(len(A)):
        if i != word_index:
            s = similarity[i]
            if s >= 0.8:
                d[words[i]] = s

    d = sort_dict(d)
    tmp = {}
    # count = 0

    # change the count of words you want in this boundry
    # limit = 18

    for key in d.keys():
        tmp[key] = d[key]
        # count += 1
        # if count >= limit:
        #     break
    return tmp


""" make connection between all the words that have similarity between 0.8 to 0.9 """


def stage_two(A, word_index, words, similarity):
    d = {}
    for i in range(len(A)):
        if i != word_index:
            s = similarity[i]
            if 0.8 > s >= 0.7:
                d[words[i]] = s

    d = sort_dict(d)
    tmp = {}
    # count = 0

    # change the count of words you want in this boundry
    # limit = 10

    for key in d.keys():
        tmp[key] = d[key]
        # count += 1
        # if count >= limit:
        #     break

    return tmp


""" make connection between all the words that have similarity between 0.7 to 0.8 """


def stage_three(A, word_index, words, similarity):
    d = {}
    for i in range(len(A)):
        if i != word_index:
            s = similarity[i]
            if 0.7 > s >= 0.6:
                d[words[i]] = s

    d = sort_dict(d)
    tmp = {}
    count = 0
    # change the count of words you want in this boundry
    # limit = 5

    for key in d.keys():
        tmp[key] = d[key]
        # count += 1
        # if count >= limit:
        #     break

    return tmp


""" make connection between all the words that have similarity between 0.7 to 0.8 """


def stage_four(A, word_index, words, similarity):
    d = {}
    for i in range(len(A)):
        if i != word_index:
            s = similarity[i]
            if 0.6 > s >= 0.5:
                d[words[i]] = s

    d = sort_dict(d)
    tmp = {}
    count = 0
    # change the count of words you want in this boundry
    # limit = 5

    for key in d.keys():
        tmp[key] = d[key]
        # count += 1
        # if count >= limit:
        #     break

    return tmp


""" make connection between all the words that have similarity between 0.7 to 0.8 """


def stage_five(A, word_index, words, similarity):
    d = {}
    for i in range(len(A)):
        if i != word_index:
            s = similarity[i]
            if 0.5 > s >= 0.36:
                d[words[i]] = s

    d = sort_dict(d)
    tmp = {}
    count = 0
    # change the count of words you want in this boundry
    # limit = 5

    for key in d.keys():
        tmp[key] = d[key]
        # count += 1
        # if count >= limit:
        #     break

    return tmp


def stages(A, word_index, words, similarity):
    l1, l2, l3, l4, l5 = [], [], [], [], []
    d = {}
    for i in range(len(A)):
        if i != word_index:
            if similarity[word_index][i] >= 0.8:
                l1.append([words[i], similarity[word_index][i]])
            elif 0.7 <= similarity[word_index][i] < 0.8:
                l2.append([words[i], similarity[word_index][i]])
            elif 0.6 <= similarity[word_index][i] < 0.7:
                l3.append([words[i], similarity[word_index][i]])
            elif 0.5 <= similarity[word_index][i] < 0.6:
                l4.append([words[i], similarity[word_index][i]])
            elif 0.36 <= similarity[word_index][i] < 0.5:
                l5.append([words[i], similarity[word_index][i]])

    l1 = check_lists(l1, 10)
    n1 = len(l1)
    if n1 < 15:
        l2 = check_lists(l2, n1)
        n2 = len(l2)
        if (n2 + n1) < 10:
            l3 = check_lists(l3, n2)
            n3 = len(l3)
            if (n3 + n2 + n1) < 10:
                l4 = check_lists(l3, n3)
                n4 = len(l4)
                if (n3 + n2 + n1 + n4) < 10:
                    l5 = check_lists(l4, n4)
                    d = create_dict(l1, l2, l3, l4, l5)
                d = create_dict(l1, l2, l3, l4, [])
            d = create_dict(l1, l2, l3, [], [])
        d = create_dict(l1, l2, [], [], [])
    d = create_dict(l1, [], [], [], [])
    return d


def create_file(d):
    N = {}
    nei = []
    for key in d.keys():
        for key1 in d[key].keys():
            nei.append(key1)
        N[key] = nei
        nei = []
    csv_columns = ['Word', 'Neighbors', 'degree']
    csv_file = "Graph_idea.csv"
    try:
        with open(csv_file, 'w', encoding='utf-8', newline='') as csvfile:
            degree = []
            writer = csv.writer(csvfile)
            writer.writerow(csv_columns)
            for key, value in N.items():
                writer.writerow([key, value, len(value)])
                degree.append(len(value))
            writer.writerow(["minimum degree", min(degree)])
            writer.writerow(["maximum degree", max(degree)])
            writer.writerow(["Avarage degree", list_avg(degree)])
    except IOError:
        print("I/O error")


def get_connection(A, words, full_A):
    d = {}
    similarity = []
    for i in range(len(A)):
        similarity.append([])
        for j in range(len(full_A)):
            similarity[i].append(cosine_similarity(A[i], full_A[j]))

    for i in range(len(A)):
        tmp = stages(full_A, i, words, similarity)
        d[words[i]] = tmp
    print("creating the file")
    create_file(d)
    return d
