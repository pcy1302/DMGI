import pdb
import sys
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
import pickle as pkl
import pdb
import sys
from nltk.corpus import stopwords
import re
import time

num_train = int(sys.argv[1])

raw_data_filename = 'dblp.txt'

top_confs = ['icml','aaai','ijcai','sigkdd','international conference on data mining','sigmod','vldb', 'icde','sigir', 'cikm','cvpr','eccv','emnlp','naacl', 'iccv', 'web search and data mining', 'world wide web conference','association for computational linguistics']

conf_dict = {'international conference on data mining':'icdm', 'web search and data mining':'wsdm',
            'world wide web conference':'www','association for computational linguistics':'acl',
            'sigkdd':'sigkdd','vldb':'vldb','sigmod':'sigmod','icde':'icde','icml':'icml',
            'sigir':'sigir','cvpr':'cvpr','cikm':'cikm','eccv':'eccv',"aaai":"aaai","emnlp":"emnlp",
            'ijcai':'ijcai','naacl':'naacl','iccv':'iccv'}

def currentTime():
    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (
        now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s

print('[{}]Reading {}'.format(currentTime(), raw_data_filename))
with open(raw_data_filename, 'r', errors='replace') as f:
    lines = f.readlines()

# separate blocks
papers = []; paper = []
for line in lines:
    if len(line.strip()) != 0: paper.append(line.rstrip())
    else: papers.append(paper); paper = []

# read papers 
titles = []; authors = []; years = []; conferences = []; indices = []; references = []; abstracts = []
for paper in papers:
    if '#*' not in str(paper) or '#@' not in str(paper) or '#t' not in str(paper) or '#c' not in str(paper) or '#!' not in str(paper): 
        continue
    ref_tmp = []; flag=True
    for elem in paper:
        if elem.startswith('#*'):
            title = elem.split("#*")[1]
            if len(title) == 0:
                flag=False
                break
            
        elif elem.startswith('#@'):
            author = elem.split("#@")[1]
            if len(author) == 0:
                flag=False
                break
            
        elif elem.startswith('#t'):
            year = elem.split("#t")[1]
            if len(year) == 0:
                flag=False
                break
            
        elif elem.startswith('#c'):
            conference = elem.split("#c")[1]
            if len(conference) == 0:
                flag=False
                break
            
        elif elem.startswith('#index'):
            index = elem.split("#index")[1]
            if len(index) == 0:
                flag=False
                break
            
        elif elem.startswith('#%'):
            reference = elem.split("#%")[1]
            ref_tmp.append(reference)
            
            
        elif elem.startswith('#!'):
            abstract = elem.split("#!")[1]
            if len(abstract) < 100:
                flag=False
                break

    if flag==True:
        titles.append(title); authors.append(author); years.append(year); conferences.append(conference); indices.append(index); references.append(ref_tmp); abstracts.append(abstract)

df = pd.DataFrame({'paper_idx':indices, 'title':titles, 'author':authors, 'year':years,'conference':conferences, 'abstract':abstracts, 'reference':references})


df.year = df.year.astype('int')
df.conference = df.conference.str.lower()
df.abstract = df.abstract.str.lower()
df.title = df.title.str.lower()
print("[{}]Done reading data".format(currentTime()))

print("[{}]Start Filtering data".format(currentTime()))

df = df.drop_duplicates(subset='title')

df_year = df.loc[(df.year >= 2006) & (df.year <= 2015)]
df_year = df_year.loc[(~df_year.conference.str.contains('workshop'))]
df_year = df_year.loc[(~df_year.conference.str.contains('special issue'))]
df_year = df_year.loc[(~df_year.conference.str.contains('tutorials'))]
df_year = df_year.loc[(~df_year.conference.str.contains('companion'))]
df_year = df_year.loc[(~df_year.conference.str.contains('poster'))]
df_year = df_year.loc[(~df_year.conference.str.contains('posters'))]
df_year = df_year.loc[(~df_year.conference.str.contains('talks'))]
df_year = df_year.loc[(~df_year.conference.str.contains('sketches'))]
df_year = df_year.loc[(~df_year.conference.str.contains('courses'))]
df_year = df_year.loc[(~df_year.conference.str.contains('classes'))]

print("[{}]Done Filtering data".format(currentTime()))

df_year = df_year[df_year.conference.str.lower().str.contains(('|'.join(top_confs)))]

conf_dict_tmp = {}
for elem in df_year.conference.unique():
    for idx, conf in enumerate(top_confs):
        if conf in elem:
            conf_dict_tmp[elem] = conf
            break

df_year.conference = df_year.conference.map(conf_dict_tmp)
df_year.conference = df_year.conference.map(conf_dict)
print("Num conferences: {}".format(len(df_year.conference.unique())))

# parse authors
df_year['author']= df_year.author.apply(lambda x: x.split(", "))

def print_stats():
    num_papers = len(df_year.paper_idx.unique())
    num_conferences = len(df_year.conference.unique())
    num_authors = len(set([elem for elems in df_year.author for elem in elems]))
    ave_authors_per_paper = df_year[['paper_idx','author']].author.apply(len).sum() / len(df_year)
    
    print("NumPapers: {}, NumVenues: {}, NumAuthors: {}, AveAuthorsPerPaper: {}"
          .format(num_papers, num_conferences, num_authors, round(ave_authors_per_paper,3)))

# print_stats()

df_filtered = df_year[['paper_idx','author','title','conference','abstract','reference']]

df_filtered = df_filtered.reset_index(drop=True)


label_dict = {"sigkdd":"DM", "wsdm":"DM","icdm":"DM",
              "icml":"AI", "aaai":"AI", "ijcai":"AI",
              "cvpr":"CV", 
              "acl":"NLP", "naacl":"NLP", "emnlp":"NLP"}
# pdb.set_trace()
df_filtered['label'] = df_filtered.conference.map(label_dict)
df_filtered = df_filtered.dropna().reset_index(drop=True)

temp = df_filtered

temp = temp[['title','paper_idx','author','conference','abstract','reference','label']]

from collections import Counter
# filter authors
author_threshold = 3
print("Author threshold: {}".format(author_threshold))
counter = Counter([author for authors in temp.author.values for author in authors])
cnt_stopauthors = set([author for author, count in counter.most_common() if count <= author_threshold])
def remove_cnt_stopauthors(x):
    return list(set(x).difference(cnt_stopauthors))
temp.author = temp.author.apply(lambda x : remove_cnt_stopauthors(x))
temp = temp[temp.author.apply(len) > 0].reset_index(drop=True)


################# preprocess abstract
from sklearn.feature_extraction.text import TfidfVectorizer
tvec = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, min_df=.0025, max_df=.1, ngram_range=(1,1), lowercase=False)
temp['abstract'] = temp.abstract.apply(lambda x : x.lower().split(" "))
tvec.fit(temp.abstract.values.tolist())
tvec_weights = tvec.transform(temp.abstract.values.tolist())
weights = np.asarray(tvec_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': tvec.get_feature_names(), 'weight': weights})
# weights_df.sort_values(by='weight', ascending=False).head(20)
valid_words = set(weights_df.sort_values(by='weight', ascending=False).head(2000).term.values)
# temp['plots'] = temp.plots.apply(lambda x:x.split(" "))
word_set = {term for terms in temp.abstract.values for term in terms if term in valid_words}
word_idx = {word:idx for idx,word in enumerate(word_set)}
word_idx_rev = {idx:word for idx,word in enumerate(word_set)}
def map_word_dict(xs, idx_dic):
    return [idx_dic[x] for x in xs if x in idx_dic]
temp['abstract'] = temp.abstract.apply(lambda x : map_word_dict(x, word_idx))
temp = temp[temp.abstract.apply(len) > 0].reset_index(drop=True)
print("num abstract words: {}".format(len(set([word for words in temp.abstract.values for word in words]))))
################





tvec = TfidfVectorizer(stop_words='english', min_df=1, max_df=1.0, ngram_range=(1,1))
tvec.fit(temp.title)
tvec_weights = tvec.transform(temp.title.values.tolist())
weights = np.asarray(tvec_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': tvec.get_feature_names(), 'weight': weights})
valid_words = set(weights_df.sort_values(by='weight', ascending=False).head(2000).term.values)

temp['title_real'] = temp.title #############
temp['title'] = temp.title.apply(lambda x:x.lower().split(" "))

title_word_set = {term for terms in temp.title.values for term in terms if term in valid_words}
title_word_idx = {word:idx for idx,word in enumerate(title_word_set)}
title_word_idx_rev = {idx:word for idx,word in enumerate(title_word_set)}

def map_word_dict(xs, idx_dic):
    return [idx_dic[x] for x in xs if x in idx_dic]
temp['title'] = temp.title.apply(lambda x : map_word_dict(x, title_word_idx))
temp = temp[temp.title.apply(len) > 0].reset_index(drop=True)
print("num title words: {}".format(len(set([word for words in temp.title.values for word in words]))))

papers = set(temp.paper_idx.unique())
# leave ref papers that are in papers
def filter_refs(x):
    return list(set(x).intersection(papers))
temp['reference'] = temp.reference.apply(lambda x: filter_refs(x))

# Map to indices
paper_idx_names = list(set([paper_idx for paper_idx in temp.paper_idx.values]))
paper_idx_names_dict = {name:idx for idx, name in enumerate(paper_idx_names)}
paper_idx_names_dict_rev = {idx:name for idx, name in enumerate(paper_idx_names)}

area_names = list(set([area for area in temp.label.values]))
area_names_dict = {name:idx for idx, name in enumerate(area_names)}
area_names_dict_rev = {idx:name for idx, name in enumerate(area_names)}

author_names = set([author for authors in temp.author.values for author in authors])
author_names_dict = {name:idx for idx, name in enumerate(author_names)}
author_names_dict_rev = {idx:name for idx, name in enumerate(author_names)}


conf_names = list(set([conf for conf in temp.conference.values]))
conf_names_dict = {name:idx for idx, name in enumerate(conf_names)}

# map label
temp.label = temp.label.map(area_names_dict)

# map paper idx
temp.paper_idx = temp.paper_idx.map(paper_idx_names_dict)

# map reference
def map_refs_dict(xs):
    return [paper_idx_names_dict[x] for x in xs]
temp.reference = temp.reference.apply(lambda x : map_refs_dict(x))

# map author
def map_author_dict(xs):
    return [author_names_dict[x] for x in xs]
temp['author_real'] = temp.author
temp.author = temp.author.apply(lambda x : map_author_dict(x))

# map conference
temp['conference_real'] = temp.conference
temp.conference = temp.conference.map(conf_names_dict)

# temp = temp[['title','paper_idx','author','conference','abstract','reference','label']]
def make_onehot(idxs, length):
    tmp = [0] * length
    for idx in idxs:
        tmp[idx] = 1
    return tmp

# PC = []
PA = []
# PP_ref = []
PT = []
features = []
labels = []
author_titles = dict()
paper_idxs = []
for idx, vals in enumerate(temp.values):
    title = vals[0]
    paper_idx = vals[1]
    authors = vals[2]
#     conf = [vals[3]]
    abstract = vals[4]
    refs = vals[5]
    area = [vals[6]]
    
#     PC.append(make_onehot(conf, len(conf_names_dict)))
    paper_idxs.append(paper_idx)
    PT.append(make_onehot(title, len(title_word_idx_rev)))
    PA.append(make_onehot(authors, len(author_names_dict)))
#     PP_ref.append(make_onehot(refs, len(paper_idx_names_dict)))    
    features.append(make_onehot(abstract, len(word_idx_rev)))
    labels.append(make_onehot(area, len(area_names_dict)))
    
    for author in authors:
        author_titles.setdefault(author,[]).extend(title)

paper_idxs_rev = {p_idx:idx for idx, p_idx in enumerate(paper_idxs)}
PP_ref = []
for idx, vals in enumerate(temp.values):
    refs = [paper_idxs_rev[elem] for elem in vals[5]]
    PP_ref.append(make_onehot(refs, len(paper_idx_names_dict)))  
    
    

AT = []    
for author in range(len(author_titles)):
    titles = list(set(author_titles[author]))
    AT.append(make_onehot(titles, len(title_word_idx_rev)))
    

PA = np.array(PA).astype(float)
PP_ref = np.array(PP_ref).astype(float)
PT = np.array(PT).astype(float)
# PC = np.array(PC).astype(float)
AT = np.array(AT).astype(float)
features = np.array(features).astype(float)
labels = np.array(labels).astype(float)
def print_shape(mat, name):
    print("[{}] shape:{} / numRelations: {}".format(name, mat.shape, len(mat.nonzero()[0])))
    
print_shape(PA, 'PA')
print_shape(PP_ref, 'PP_ref')
print_shape(PT, 'PT')
print_shape(AT, 'AT')
print_shape(features, 'features')

PAP = np.matmul(PA, PA.T) #

# PCP = np.matmul(PC, PC.T) #
PAT = np.matmul(PA, AT) 
PATA = np.matmul(PAT, AT.T)
PATAP = np.matmul(PATA, PA.T) #


PPrefP = np.matmul(PP_ref, PP_ref.T)
PTP = np.matmul(PT, PT.T)

PAP = (PAP > 0) * np.ones_like(PAP)
# PCP = (PCP > 0) * np.ones_like(PCP)
PATAP = (PATAP > 0) * np.ones_like(PATAP)
PPrefP = (PPrefP > 0) * np.ones_like(PPrefP)
PTP = (PTP > 0) * np.ones_like(PTP)

arg_labels = np.argmax(labels,1)
unique, counts = np.unique(arg_labels, return_counts=True)
print("Label: {}".format({area_names_dict_rev[un]:cn for un, cn in zip(unique, counts)}))

label_idxs_dict = {}
for idx, label in enumerate(arg_labels):
    label_idxs_dict.setdefault(label, []).append(idx)

train_idx = []
val_idx = []
test_idx = []
for label in label_idxs_dict:
    idxs = label_idxs_dict[label]
    train_idx += idxs[:num_train]
    val_idx += idxs[num_train:num_train+50]
    test_idx += idxs[num_train+50:]

print("Train: {}, Val: {}, Test: {}".format(len(train_idx), len(val_idx), len(test_idx)))
train_idx = np.array(train_idx).reshape(1,-1)
val_idx = np.array(val_idx).reshape(1,-1)
test_idx = np.array(test_idx).reshape(1,-1)

data = {'label':labels, 'feature':features, 'PAP':PAP, 'PPP':PPrefP, 'PATAP':PATAP, 'train_idx':train_idx, 'val_idx':val_idx, 'test_idx':test_idx}


print('dblp_num_labels_{}.pkl'.format(num_train))
pkl.dump(data, open('dblp_{}.pkl'.format(num_train),"wb"), protocol=4)

def print_sparsity(mat, name):
    print("[{}] Density: {}".format(name, np.mean(sum(mat==1) / mat.shape[1])))
    
print_sparsity(PAP,'PAP')
print_sparsity(PPrefP,'PPP')
print_sparsity(PATAP,'PATAP')

def print_shape(mat, name):
    print("[{}] shape:{} / numRelations: {}".format(name, mat.shape, len(mat.nonzero()[0])))
    

print_shape(PAP,'PAP')
print_shape(PPrefP,'PPP')
print_shape(PATAP,'PATAP')    

