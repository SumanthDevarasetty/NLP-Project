from textblob import TextBlob
import wikipedia
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import numpy as np
import json
import argparse
import math
import numpy as np

# Create an argument parser
parser = argparse.ArgumentParser(description='main.py')

# Tunable parameters as external arguments
parser.add_argument('-dataset', default = "cranfield/", 
          help = "Path to the dataset folder")
parser.add_argument('-out_folder', default = "output/", 
          help = "Path to output folder")
parser.add_argument('-custom', action = "store_true", 
          help = "Take custom query as input")

# Parse the input arguments
args = parser.parse_args()

def queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k):
  """
  Computation of precision of the Information Retrieval System
  at a given value of k for a single query

  Parameters
  ----------
  arg1 : list
    A list of integers denoting the IDs of documents in
    their predicted order of relevance to a query
  arg2 : int
    The ID of the query in question
  arg3 : list
    The list of IDs of documents relevant to the query (ground truth)
  arg4 : int
    The k value

  Returns
  -------
  float
    The precision value as a number between 0 and 1
  """

  precision = -1

  #Fill in code here
  retrieved_k = query_doc_IDs_ordered[0:k]
  rel_and_ret = 0
  for doc_ID in retrieved_k:
    if doc_ID in true_doc_IDs:
      rel_and_ret = rel_and_ret + 1
  precision = rel_and_ret*1.0/k

  return precision


def meanPrecision(doc_IDs_ordered, query_ids, qrels, k):
  """
  Computation of precision of the Information Retrieval System
  at a given value of k, averaged over all the queries

  Parameters
  ----------
  arg1 : list
    A list of lists of integers where the ith sub-list is a list of IDs
    of documents in their predicted order of relevance to the ith query
  arg2 : list
    A list of IDs of the queries for which the documents are ordered
  arg3 : list
    A list of dictionaries containing document-relevance
    judgements - Refer cran_qrels.json for the structure of each
    dictionary
  arg4 : int
    The k value

  Returns
  -------
  float
    The mean precision value as a number between 0 and 1
  """

  meanPrecision = -1

  #Fill in code here
  precision_sum = 0

  for i in range(len(query_ids)):
    true_doc_IDs = []
    for j in qrels:
      if int(j["query_num"]) == query_ids[i]:
        true_doc_IDs.append(int(j["id"]))
    precision_sum = precision_sum + queryPrecision(doc_IDs_ordered[i], query_ids[i], true_doc_IDs, k)
  meanPrecision = precision_sum/len(query_ids)

  return meanPrecision


def queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k):
  """
  Computation of recall of the Information Retrieval System
  at a given value of k for a single query

  Parameters
  ----------
  arg1 : list
    A list of integers denoting the IDs of documents in
    their predicted order of relevance to a query
  arg2 : int
    The ID of the query in question
  arg3 : list
    The list of IDs of documents relevant to the query (ground truth)
  arg4 : int
    The k value

  Returns
  -------
  float
    The recall value as a number between 0 and 1
  """

  recall = -1

  #Fill in code here
  retrieved_k = query_doc_IDs_ordered[0:k]
  rel_and_ret = 0
  for doc_ID in retrieved_k:
    if doc_ID in true_doc_IDs:
      rel_and_ret = rel_and_ret + 1
  recall = rel_and_ret*1.0/len(true_doc_IDs)
  return recall


def meanRecall(doc_IDs_ordered, query_ids, qrels, k):
  """
  Computation of recall of the Information Retrieval System
  at a given value of k, averaged over all the queries

  Parameters
  ----------
  arg1 : list
    A list of lists of integers where the ith sub-list is a list of IDs
    of documents in their predicted order of relevance to the ith query
  arg2 : list
    A list of IDs of the queries for which the documents are ordered
  arg3 : list
    A list of dictionaries containing document-relevance
    judgements - Refer cran_qrels.json for the structure of each
    dictionary
  arg4 : int
    The k value

  Returns
  -------
  float
    The mean recall value as a number between 0 and 1
  """

  meanRecall = -1

  #Fill in code here
  recall_sum = 0

  for i in range(len(query_ids)):
    true_doc_IDs = []
    for j in qrels:
      if int(j["query_num"]) == query_ids[i]:
        true_doc_IDs.append(int(j["id"]))
    recall_sum = recall_sum + queryRecall(doc_IDs_ordered[i], query_ids[i], true_doc_IDs, k)
  meanRecall = recall_sum/len(query_ids)

  return meanRecall


def queryFscore(query_doc_IDs_ordered, query_id, true_doc_IDs, k):
  """
  Computation of fscore of the Information Retrieval System
  at a given value of k for a single query

  Parameters
  ----------
  arg1 : list
    A list of integers denoting the IDs of documents in
    their predicted order of relevance to a query
  arg2 : int
    The ID of the query in question
  arg3 : list
    The list of IDs of documents relevant to the query (ground truth)
  arg4 : int
    The k value

  Returns
  -------
  float
    The fscore value as a number between 0 and 1
  """

  fscore = -1

  #Fill in code here
  retrieved_k = query_doc_IDs_ordered[0:k]
  rel_and_ret = 0
  for doc_ID in retrieved_k:
    if doc_ID in true_doc_IDs:
      rel_and_ret = rel_and_ret + 1
  precision = rel_and_ret*1.0/k
  recall = rel_and_ret*1.0/len(true_doc_IDs)

  if precision+recall == 0:
    fscore = 0
  else:
    fscore = (2*precision*recall)/(precision+recall)
  return fscore


def meanFscore(doc_IDs_ordered, query_ids, qrels, k):
  """
  Computation of fscore of the Information Retrieval System
  at a given value of k, averaged over all the queries

  Parameters
  ----------
  arg1 : list
    A list of lists of integers where the ith sub-list is a list of IDs
    of documents in their predicted order of relevance to the ith query
  arg2 : list
    A list of IDs of the queries for which the documents are ordered
  arg3 : list
    A list of dictionaries containing document-relevance
    judgements - Refer cran_qrels.json for the structure of each
    dictionary
  arg4 : int
    The k value
  
  Returns
  -------
  float
    The mean fscore value as a number between 0 and 1
  """

  meanFscore = -1

  #Fill in code here
  Fscore_sum = 0

  for i in range(len(query_ids)):
    true_doc_IDs = []
    for j in qrels:
      if int(j["query_num"]) == query_ids[i]:
        true_doc_IDs.append(int(j["id"]))
    Fscore_sum = Fscore_sum + queryFscore(doc_IDs_ordered[i], query_ids[i], true_doc_IDs, k)

  meanFscore = Fscore_sum/len(query_ids)
  return meanFscore

def queryNDCG(query_doc_IDs_ordered, query_id, true_doc_IDs, k):
  """
      Computation of nDCG of the Information Retrieval System
      at given value of k for a single query

      Parameters
      ----------
      arg1 : list
          A list of integers denoting the IDs of documents in
          their predicted order of relevance to a query
      arg2 : int
          The ID of the query in question
      arg3 : list
          The list of IDs of documents relevant to the query (ground truth)
      arg4 : int
          The k value

      Returns
      -------
      float
          The nDCG value as a number between 0 and 1
      """

  nDCG = -1

  # Fill in code here
  DCG = 0
  IDCG = 0
  retrieved_k = query_doc_IDs_ordered[0:k]
  true_docs_ids = np.array(true_doc_IDs)[:,0]
  true_docs_rel = np.array(true_doc_IDs)[:,1]

  ret_docs_rel = []

  for j in range(len(retrieved_k)):
    for i in range(len(true_docs_ids)):
      if retrieved_k[j] == true_docs_ids[i]:
        ret_docs_rel.append(true_docs_rel[i])
        DCG = DCG + true_docs_rel[i]*1.0/math.log(j+2,2)

  rel_sorted = np.sort(true_docs_rel)[::-1][0:k]
  for i in range(len(rel_sorted)):
    IDCG = IDCG + (rel_sorted[i] * 1.0 / math.log(i + 2, 2))
  if IDCG == 0:
    nDCG = 0
  else:
    nDCG = DCG/IDCG
  return nDCG

def meanNDCG(doc_IDs_ordered, query_ids, qrels, k):
  """
      Computation of nDCG of the Information Retrieval System
      at a given value of k, averaged over all the queries

      Parameters
      ----------
      arg1 : list
          A list of lists of integers where the ith sub-list is a list of IDs
          of documents in their predicted order of relevance to the ith query
      arg2 : list
          A list of IDs of the queries for which the documents are ordered
      arg3 : list
          A list of dictionaries containing document-relevance
          judgements - Refer cran_qrels.json for the structure of each
          dictionary
      arg4 : int
          The k value

      Returns
      -------
      float
          The mean nDCG value as a number between 0 and 1
      """

  meanNDCG = -1

  # Fill in code here
  NDCG_sum = 0

  for i in range(len(query_ids)):
    true_doc_IDs = []
    for j in qrels:
      if int(j["query_num"]) == query_ids[i]:
        true_doc_IDs.append([int(j["id"]), 5 - j["position"]])
    NDCG_sum = NDCG_sum + queryNDCG(doc_IDs_ordered[i], query_ids[i], true_doc_IDs, k)

  meanNDCG = NDCG_sum / len(query_ids)

  return meanNDCG

def queryAveragePrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k):
  """
  Computation of average precision of the Information Retrieval System
  at a given value of k for a single query (the average of precision@i
  values for i such that the ith document is truly relevant)

  Parameters
  ----------
  arg1 : list
    A list of integers denoting the IDs of documents in
    their predicted order of relevance to a query
  arg2 : int
    The ID of the query in question
  arg3 : list
    The list of documents relevant to the query (ground truth)
  arg4 : int
    The k value

  Returns
  -------
  float
    The average precision value as a number between 0 and 1
  """

  avgPrecision = -1

  #Fill in code here
  retrieved_k = query_doc_IDs_ordered[0:k]
  precision_sum = 0
  count = 0
  for l in range(k):
    if retrieved_k[l] in true_doc_IDs:
      precision_sum = precision_sum + queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, l+1)
      count = count + 1
  if count == 0:
    avgPrecision = 0
  else:
    avgPrecision = precision_sum*1.0/count
  return avgPrecision


def meanAveragePrecision(doc_IDs_ordered, query_ids, q_rels, k):
  """
  Computation of MAP of the Information Retrieval System
  at given value of k, averaged over all the queries

  Parameters
  ----------
  arg1 : list
    A list of lists of integers where the ith sub-list is a list of IDs
    of documents in their predicted order of relevance to the ith query
  arg2 : list
    A list of IDs of the queries
  arg3 : list
    A list of dictionaries containing document-relevance
    judgements - Refer cran_qrels.json for the structure of each
    dictionary
  arg4 : int
    The k value

  Returns
  -------
  float
    The MAP value as a number between 0 and 1
  """

  meanAveragePrecision = -1


  #Fill in code here
  AP_sum = 0

  for i in range(len(query_ids)):
    true_doc_IDs = []
    for j in q_rels:
      if int(j["query_num"]) == query_ids[i]:
        true_doc_IDs.append(int(j["id"]))
    AP_sum = AP_sum + queryAveragePrecision(doc_IDs_ordered[i], query_ids[i], true_doc_IDs, k)

  meanAveragePrecision = AP_sum/len(query_ids)

  return meanAveragePrecision


stemmer = SnowballStemmer(language='english')
lemmatizer = WordNetLemmatizer()

# Cleaning and Tokenizing cranfield data 
stop_words = stopwords.words("english") + ['.',',',')','(']
tokenizedText=[]
tokenized_Text=[]
stopwordRemovedText=[]
vocab = []
docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
doc_ids, docs, titles = [item["id"] for item in docs_json], \
							[item["body"] for item in docs_json], \
              [item["title"] for item in docs_json]


for doc in docs:
  tokenizedText.append(TreebankWordTokenizer().tokenize(doc))

for doc in tokenizedText:
  filtered_sentence = []
  for w in doc:
    #w = lemmatizer.lemmatize(w)
    w = stemmer.stem(w)
    if w not in stop_words:
      filtered_sentence.append(w)
      if w not in vocab:
        vocab.append(w)
  stopwordRemovedText.append(filtered_sentence)

vocab_len = len(vocab)
concept_len3 = len(stopwordRemovedText)
tf_matrix3=np.zeros([vocab_len,concept_len3])

for j in range(concept_len3):
  for w in stopwordRemovedText[j]:
    if w in vocab:
      i = vocab.index(w)
      tf_matrix3[i][j]+=1

import math
tf_idf_matrix3 = tf_matrix3.copy()
for i in range(vocab_len):
    #tf_idf_matrix[i] =  tf_idf_matrix[i] * math.log(concept_len+vocab_len/(np.count_nonzero(tf_matrix[i])+1))
  tf_idf_matrix3[i] =  tf_idf_matrix3[i] * math.log(concept_len3/(np.count_nonzero(tf_matrix3[i])+1),10)

import numpy as np
K = 300
# Applying SVD
U, s, VT = np.linalg.svd(tf_idf_matrix3) # .T is used to take transpose and .toarray() is used to convert sparse matrix to normal matrix

TF_IDF_matrix_reduced = np.dot(U[:,:K], np.dot(np.diag(s[:K]), VT[:K, :]))

# Getting document and term representation
terms_rep = np.dot(U[:,:K], np.diag(s[:K])) # M X K matrix where M = Vocabulary Size and N = Number of documents
docs_rep = np.dot(np.diag(s[:K]), VT[:K, :]).T # N x K matrix 
   
def query(stri):
  query_vector = np.zeros(K)
  for w in TreebankWordTokenizer().tokenize(stri):
    #w = lemmatizer.lemmatize(w)
    w = stemmer.stem(w)
    if w in vocab:
      i = vocab.index(w)
      query_vector = query_vector+(terms_rep[i])* math.log(concept_len3/(np.count_nonzero(tf_matrix3[i])+1),10)
  dot_vec = np.dot(docs_rep,query_vector)
  cos_sim = np.zeros(n)
  for i in range(n):
    if np.linalg.norm(docs_rep[i]) == 0:
      cos_sim[i] = 0
    else:
      if np.linalg.norm(query_vector) == 0:
        cos_sim[i] = 0
        print('bad query')
      else:
        cos_sim[i] = dot_vec[i] * (1/np.linalg.norm(docs_rep[i])) * (1/np.linalg.norm(query_vector))
        #cos_sim[i] = dot_vec[i]
  sorted_sim = np.argsort(cos_sim)[::-1]
  sorted_sim = [i+1 for i in sorted_sim ]
  return sorted_sim

n = len(stopwordRemovedText)

if args.custom:
  """
  Take a custom query as input and return top five relevant documents
  """

  #Get query
  print("Enter query below")
  query_ = input()

  doc_IDs_ordered = query(query_)

  # Print the IDs of first five documents
  print("\nTop five document IDs : ")
  for id_ in doc_IDs_ordered[:5]:
    print(id_)

else: 
  queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
  query_ids, queries = [item["query number"] for item in queries_json], \
                  [item["query"] for item in queries_json]
  qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]
  doc_IDs_ordered = np.zeros([len(queries),n])
  for i in range(len(queries)):
    doc_IDs_ordered[i] = query(queries[i])

  precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
  for k in range(1, 11):
    precision = meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
    precisions.append(precision)
    recall = meanRecall(doc_IDs_ordered, query_ids, qrels, k)
    recalls.append(recall)
    fscore = meanFscore(doc_IDs_ordered, query_ids, qrels, k)
    fscores.append(fscore)
    print("Precision, Recall and F-score @ " +  str(k) + " : " + str(precision) + ", " + str(recall) +  ", " + str(fscore))
    MAP = meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
    MAPs.append(MAP)
    nDCG = meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
    nDCGs.append(nDCG)
    print("MAP, nDCG @ " +  str(k) + " : " + str(MAP) + ", " + str(nDCG))

  import matplotlib.pyplot as plt
  plt.plot(range(1, 11), precisions, label="Precision")
  plt.plot(range(1, 11), recalls, label="Recall")
  plt.plot(range(1, 11), fscores, label="F-Score")
  plt.plot(range(1, 11), MAPs, label="MAP")
  plt.plot(range(1, 11), nDCGs, label="nDCG")
  plt.legend()
  plt.title("Evaluation Metrics - Cranfield Dataset")
  plt.xlabel("k")
  plt.savefig(args.out_folder+"eval_plot.png")
  print("Plot saved to working Directory")


