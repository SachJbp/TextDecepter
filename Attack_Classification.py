import nltk
import numpy as np
import tensorflow_datasets as tfds
import pattern
import time
import criteria
import torch
import dataloader
import argparse
import os
import re
from train_classifier import Model
import random
import tensorflow as tf
import tensorflow_hub as hub
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from itertools import combinations
from nltk.tokenize import RegexpTokenizer
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Semantic_Sim import semantic_sim
from Synonym_Picker import pick_most_similar_words_batch
from download_attack_data import download_attack_data
from compute_cos_sim_mat import compute_cos_sim_mat
from generate_embedding_mat import generate_embedding_mat
import LoadPretrainedBert
import generateCosSimMat
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from collections import defaultdict

nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')


class model:
  '''Class to Encapsulate Text Classification model usage 
  '''
  
  def __init__(self,cmodel = None, gcp = False , gcp_nlp_json_link = "" ):
    '''Constructs all the necessary attributes for the Model object.
    
    Args:
      cmodel (obj): Text Classification model object
      gcp (bool):  Whether to attack Google Cloud Platform (GCP) Text classification model
      gcp_nlp_json_link (str): GCP NLP Key Json file link

    '''
    self.cmodel=cmodel
    self.gcp = gcp
    self.gcp_key = gcp_nlp_json_link
      
  def getPredictions(self, revs):
    '''Prediction function for Text classsification task

    Args:
      revs (list): List of reviews

    '''
    if not self.gcp:
      revs1=[rev.split() for rev in revs]
      orig_probs = self.cmodel.text_pred(revs1).squeeze()
      #print(orig_probs)
      if len(revs)>1:
        orig_label = torch.argmax(orig_probs,axis=1)
      else:
        orig_label = torch.argmax(orig_probs)
        return [orig_label.tolist()]
      
      return orig_label.tolist()
    
    else:
      os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.gcp_key
      client = language.LanguageServiceClient()
      preds=[]
      for rev in revs:
        flag=0
        document = types.Document(
            content=rev,
            type=enums.Document.Type.PLAIN_TEXT)
        while flag==0:
          try:
            sentiment = client.analyze_sentiment(document=document).document_sentiment
            preds.append(int(sentiment.score>0))  
            flag=1
          except:
            time.sleep(63)
        
        return preds
            

def get_sentence_imp_ranking(sents_sentiment_dic , num_queries, orig_label, tmodel):
  '''Computes importance ranking of sentences in the text
  
  Args:
    sents_sentiment_dic (dict): Dictionary having sentences as keys and their predicted class as values
    num_queries (int):  tracks the cumulative number of queries to the attack system
    orig_label (int): Original Predicted label of the text 
      
  '''
  
  orig_label_sents_list = sents_sentiment_dic[orig_label][:]
  other_label_sents_str_list = [' '.join(sents_sentiment_dic[lab]) for lab in sents_sentiment_dic.keys() if lab!=orig_label]
      
  imp_level = 0
  top_sent_imp = {}
  word_agg_dic = defaultdict(list)
  sent_agg_dic = {}
  tokenizer = RegexpTokenizer(r'\w+')

  while len(orig_label_sents_list) != 0:
    imp_level += 1
    top_sent_imp[imp_level] = []

    if imp_level <= len(orig_label_sents_list) and other_label_sents_str_list:
      for orig_label_sent_comb in combinations(orig_label_sents_list,imp_level):
        if num_queries >= 5000:
          break
        orig_label_sents_str = ' '.join(list(orig_label_sent_comb))
        aggregate_list = [other_label_sents_str + " " + orig_label_sents_str for other_label_sents_str in other_label_sents_str_list]      
        aggregate_preds = tmodel.getPredictions(aggregate_list)
        num_queries += len(aggregate_list)

        if orig_label in aggregate_preds:
          top_sent_imp[imp_level].extend(list(orig_label_sent_comb))
          conv_set = set(top_sent_imp[imp_level])
          top_sent_imp[imp_level] = list(conv_set)
          orig_label_aggs = [aggregate_list[i] for i in range(len(aggregate_list)) if aggregate_preds[i] == orig_label]
          
          for orig_label_agg in orig_label_aggs:
              text_tokens = tokenizer.tokenize(orig_label_agg)         
              for word in text_tokens:
                if len(word) > 1:
                  word_agg_dic[word].append(orig_label_agg)

      if len(top_sent_imp[imp_level])!=0:
          for sent in top_sent_imp[imp_level]:
            if sent in orig_label_sents_list:
                 del orig_label_sents_list[orig_label_sents_list.index(sent)]
    else:
      top_sent_imp[imp_level].extend(list(orig_label_sents_list))
      orig_label_sents_list = []
      break

  #create sent2imp dictionary
  sent2imp = {}
  for key in top_sent_imp:
    for snt in top_sent_imp[key]:
      sent2imp[snt] = key
  
  return top_sent_imp , word_agg_dic , sent2imp ,num_queries

def get_word_imp(origSents , orig_label_sents, sent2imp, sent2cluster): #revise name of sent2sent
  '''Computes word importance scores

  Args:
    origSents (list): List of sentences in original text
    orig_label_sents (list): List of sentences having predicted label same as the original label of the text
    sent2imp (dict): Dictionary mapping sentences to their importance ranking
    sent2cluster (dict):  Dictionary mapping origSents to the sentence cluster to which it belongs ( We merge sentence in orig_sents
                to reduce length of origSents, in case total sentences are above 12
  '''
  
  ind_count = 0
  import_scores = []
  nlp = spacy.load('en')
  
  for sent in origSents:
    
    if not " " in str(sent):
      text_sent = [str(sent)]
    else:
      text_tokens = nlp(sent)
      text_sent = [str(word) for word in text_tokens]

    if not sent in orig_label_sents:
      import_scores.extend([300]*len(text_sent))
    else:                                 
      pos_tags = criteria.get_pos(text_sent)
      
      if sent in sent2cluster:
        sent_imp = sent2imp[sent2cluster[sent]]
      else:
        sent_imp = sent2imp[sent]

      for i1 in range(len(text_sent)):
        if pos_tags[i1] == 'ADV':
          import_scores.append(sent_imp + 15)
        elif pos_tags[i1] == 'VERB':
          import_scores.append(sent_imp + 15)
        elif pos_tags[i1] == 'ADJ':
          import_scores.append(sent_imp)
        else:
          import_scores.append(sent_imp+50)
                  
  import_scores = np.array(import_scores)

  return import_scores

def vowel_correction(txt , idx):
   '''Grammer correction after word replacement

   Corrects the grammer in case the word replacement involves changing from a word starting with a consonant 
   to the one starting vowel or vice versa

   Args:
    txt: Str, Given text
    idx: Index before which the correction has to be done  
   
   '''
   
   vowel = {'a','e','i','o','u'}

   if txt[idx][0].lower() in vowel and idx > 0:
     if txt[idx-1] == 'a':
       txt[idx-1] = 'an'
   elif idx > 0:
     if txt[idx-1] == 'an':
       txt[idx-1] = 'a'
   
   return txt


def get_semantic_sim_window(idx , len_text ,sim_score_window):
  '''Returns semantic similarity window

  Args:
    idx: index wrt which window needs to be returned
    len_text: Number of words in original text
    sim_score_window: Length of similarity score window
  
  '''
  half_sim_score_window = (sim_score_window - 1) // 2

  if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
    text_range_min = idx - half_sim_score_window
    text_range_max = idx + half_sim_score_window + 1
  elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
    text_range_min = 0
    text_range_max = sim_score_window
  elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
    text_range_min = len_text - sim_score_window
    text_range_max = len_text
  else:
    text_range_min = 0
    text_range_max = len_text

  return text_range_min, text_range_max


def attack(cmodel, gcp_nlp_json_link, text_ls, true_label, stop_words_set, word2idx_rev, idx2word_rev, idx2word_vocab, cos_sim, pos_filter, sim_score_threshold=0.5, 
           sim_score_window=15, synonym_num=80,syn_sim=0.65):
  '''Attack function

  Implementation of the attack algorithm 
  
  Args:
    text_ls (str): Text to be attacked
    true_label (int): True class of text_ls
    cmodel (Model obj): Model to be attacked
    cos_sim (numpy array): numpy array, precomuted cosine similarity square matrix
    word2idx (dict): Mapping words to indices in the precomuted cosine similarity square matrix
    idx2word (dict): Mapping indices of precomuted cosine similarity square matrix back to words
    sim_score_threshold (float): Semantic similarity threshold while selecting or rejecting synonyms,default:0.5
    sim_score_window (int): Window size for computing semantic similarity between actual and perturbed text around the perturbed word
    synonym_num (int): Max number of candidate synonyms to be analysed
    syn_sim (float): Threshold for cosine similarity between candidate synonyms and original word,defualt:0.75 
    gcp_nlp_json_link (str) : Google Cloud Platform NLP API JSON key file link
  '''
  
  if gcp_nlp_json_link:
    tmodel = model(gcp = True, gcp_nlp_json_link = gcp_nlp_json_link)
  else:
    tmodel = model(cmodel)
      
  text_temp = text_ls[:]
  orig_label = tmodel.getPredictions([text_ls])[0]

  if true_label != orig_label:
    return '', 0, orig_label, orig_label, 0
  else:
    nlp = spacy.load('en')
    doc = nlp(str(text_ls))
    text_ls = [str(j) for j in doc]
    len_text = len(text_ls)
    if len_text < sim_score_window:
      sim_score_threshold = 0.1  # shut down the similarity thresholding function
    num_queries = 1
    
    # get the pos info
    if pos_filter == 'fine':
      pos_ls1 = nltk.pos_tag(text_ls)
      pos_ls = [pos_ls1[i][1] for i in range(len(text_ls))] 
    else:
      pos_ls = criteria.get_pos(text_ls)
    
    #sentence segmentation
    sents_sentiment_dic = defaultdict(list)
    text_sentences = nlp(text_temp)
    sents1 = text_sentences.sents
    sents = [str(sent) for sent in sents1]
  
    #print(sents) #look up spacy to get a logical way to get phrases or chunks       
    if len(sents) == 1:
      sent = sents[0]
      tokens = nlp(sent)
      a = len(tokens)//2
      if len(tokens) > 4:
        sents = [str(tokens[i:i+4]) for i in range(0,len(tokens),4)]

    #segregate positive and negative sentence
    preds = tmodel.getPredictions(list(sents))
    num_queries += len(sents)
    for i in range(len(preds)):
        sents_sentiment_dic[preds[i]].append(sents[i])
  
    #print(sents_sentiment_dic)
    orig_sents_ln = len(sents)
    origSents = sents[:]
    orig_label_sents = sents_sentiment_dic[orig_label][:]
    
    #curtail orig label sentences
    sent2cluster = {}
    if len(orig_label_sents) > 12:
      ln = len(orig_label_sents)
      mult = int(np.ceil(ln/12))
      new_list=[]
      for q in range(0,ln,mult):
        if q+mult < ln:
          new_sent_list = sents_sentiment_dic[orig_label][q:q+mult]
          new_sent_str = ' '.join(new_sent_list)
          new_list.append(new_sent_str)
        else:
          new_sent_list=sents_sentiment_dic[orig_label][q:]
          new_sent_str = ' '.join(new_sent_list)
          new_list.append(new_sent_str)

        for snt in new_sent_list:
          sent2cluster[snt] = new_sent_str
          sents.remove(snt)

        sents.append(new_sent_str)
      sents_sentiment_dic[orig_label] = new_list
        
    #Get sentence importance ranking
    top_sent_imp , word_agg_dic ,sent2imp, num_queries = get_sentence_imp_ranking(sents_sentiment_dic , num_queries, orig_label, tmodel)
    
    #Get word importance scores            
    import_scores = get_word_imp(origSents , orig_label_sents, sent2imp, sent2cluster)

    # get words to perturb ranked by importance score for word in words_perturb
    words_perturb = []
    text_prime = text_ls[:]
    imp_indxs = np.argsort(import_scores).tolist()

    for idx in imp_indxs:
      if not text_prime[idx] in stop_words_set:
        words_perturb.append((idx, text_prime[idx]))

    # find synonyms
    words_perturb_idx = [word2idx_rev[word] for idx, word in words_perturb if word in word2idx_rev]
    synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word_vocab, synonym_num, syn_sim)
    synonyms_all = []
    for idx, word in words_perturb:
      if word in word2idx_rev:
        synonyms = synonym_words.pop(0)
        if synonyms:
          synonyms_all.append((idx, synonyms))

    # start replacing and attacking
    num_changed = 0
    idx_flag = 0
    backtrack_dic = {}
    flg = 0
    misclassified = False
    visited = {}
    
    #map the words to indices in text_prime
    word_idx_dic = {}
    for idx in range(len(text_prime)):
      word=text_prime[idx]
      if word in word_idx_dic:
        word_idx_dic[word].append(idx)
      else:
        word_idx_dic[word] = [idx]
      visited[word] = False
    
    origtext_prime = text_prime.copy()
    len_text = len(text_prime)


    for idx, synonyms in synonyms_all:

      orig_pos=criteria.get_pos(text_prime)[idx]
      
      if len(origtext_prime[idx])<=1 or visited[origtext_prime[idx]]==True:
          continue
          
      if misclassified:

        #backtrack to check for unnecessary perturbations

        for (wrd,index) in backtrack_dic:
          txt_temp = text_prime[:]
          txt_temp[index] = backtrack_dic[(wrd,index)]
          txt_temp = vowel_correction(txt_temp ,index)
          pred = tmodel.getPredictions([' '.join(txt_temp)])[0]
          num_queries += 1

          if pred!=orig_label:
            text_prime = txt_temp[:]
            num_changed-=1
        break

      if num_queries>=5000:
        break
      text_range_min, text_range_max = get_semantic_sim_window(idx, len_text,sim_score_window)

      # Step#1: Find all aggregates(with orig_label) to which the target wrd belongs
      
      target_word = text_prime[idx]
      visited[target_word] = True

      agg_list = []
      if target_word in word_agg_dic:
        agg_list = list(set(word_agg_dic[target_word]))
        word_agg_dic[target_word] = []

      orig_sentiment_sent = []
      for sent1 in sents_sentiment_dic[orig_label]:
        if target_word in sent1 and not sent1 in agg_list:
          orig_sentiment_sent.append(sent1)
        
      #Check if any synonym is able make the entire review/text misclassify
      if pos_filter == 'fine':
        new_pos = np.array([nltk.pos_tag(text_prime[:idx]+[syn]+text_prime[idx+1:])[idx][1] for syn in synonyms])
      else:
        new_pos = np.array([criteria.get_pos(text_prime[:idx]+[syn]+text_prime[idx+1:])[idx] for syn in synonyms])
      pos_mask = (new_pos==(pos_ls[idx])).astype(int)
      
      rev_with_syns1 = [text_prime[:idx]+[syn]+text_prime[idx+1:] for syn in synonyms]
      sem_sims1 = np.array([semantic_sim([' '.join(rev_with_syn[text_range_min:text_range_max])],[' '.join(text_prime[text_range_min:text_range_max])])
                    for rev_with_syn in rev_with_syns1])
      sem_sim_mask = (sem_sims1>=sim_score_threshold).astype(int)
  
      #apply pos and semantic similarity masks to synonyms
      synonyms_masked = [synonyms[i] for i in range(len(synonyms)) if pos_mask[i]==1 and sem_sim_mask[i]==1]

      rev_with_syns = [text_prime[:idx]+[syn]+text_prime[idx+1:] for syn in synonyms_masked]
      sem_sims = np.array([semantic_sim([' '.join(rev_with_syn[text_range_min:text_range_max])],[' '.join(text_prime[text_range_min:text_range_max])])
      for rev_with_syn in rev_with_syns])
      
      #sort synonyms as per semantic similarity scores
      sort_order = dict(zip(synonyms_masked,sem_sims))
      synonyms_sorted = sorted(synonyms_masked,key=sort_order.get)
      
      rev_str = ' '.join(text_prime)
      vowels ={'a','e','i','o','u'}

      revs_with_synonyms1 = [re.sub(r'\b{}\s+{}\b'.format('a',target_word),'an '+ syn , rev_str)  
                            if syn[0] in vowels else 
                            re.sub(r'\b{}\s+{}\b'.format('an',target_word),'a '+ syn, rev_str) 
                            for syn in synonyms_sorted   ]
      
      revs_with_synonyms = [re.sub(r'\b{}\b'.format(target_word),synonyms_sorted[i],revs_with_synonyms1[i])   
                            for i in range(len(synonyms_sorted))]
      
      changed=False

      for i in range(len(revs_with_synonyms)):
        num_queries += 1
        pred = tmodel.getPredictions([revs_with_synonyms[i]])[0]
        if pred != orig_label:
          changed = True
          sel_sym = synonyms_sorted[i]
          print(sel_sym)
          misclassified=True
          break
      
      #Check if any synonym is able make the any sentence, which originally had the same label as the orig_label of the review,
      #to misclassify
  
      if not changed and len(orig_sentiment_sent) > 0:
        #print("len sents: ",len(orig_sentiment_sent))
      
        sents_with_syns1 = [[re.sub(r'\b{}\s+{}\b'.format('a',target_word),'an '+ syn,sent)  
                            if syn[0] in vowels else 
                            re.sub(r'\b{}\s+{}\b'.format('an',target_word),'a '+ syn,sent) 
                            for sent in orig_sentiment_sent ] 
                            for syn in synonyms_sorted   ]
      
        sents_with_syns = [[re.sub(r'\b{}\b'.format(target_word),synonyms_sorted[i],sent)
                            for sent in sents_with_syns1[i] ]
                            for i in range(len(synonyms_sorted))]
        
        for i in range(len(sents_with_syns)):
          num_queries += len(sents_with_syns[i])
          if tmodel.getPredictions(sents_with_syns[i]).count(orig_label) < len(sents_with_syns[i]):
            changed = True
            sel_sym = synonyms_sorted[i]
            break
      
      if not changed and len(agg_list) > 0:
          
        #Check if any synonym is able make any aggregate, which originally had the same label as the orig_label of the review,
        #to misclassify

        aggs_with_syns1 = [[re.sub(r'\b{}\s+{}\b'.format('a',target_word),'an '+ syn,agg)  
                            if syn[0] in vowels else 
                            re.sub(r'\b{}\s+{}\b'.format('an',target_word),'a '+ syn,agg) 
                            for agg in agg_list ] 
                            for syn in synonyms_sorted   ]
      
        aggs_with_syns = [[re.sub(r'\b{}\b'.format(target_word),synonyms_sorted[i],agg)
                            for agg in aggs_with_syns1[i] ]
                            for i in range(len(synonyms_sorted))] 

        for i in range(len(synonyms_sorted)):
          num_queries += len(agg_list)
          if tmodel.getPredictions(aggs_with_syns[i]).count(orig_label)<len(aggs_with_syns[i]):
            changed = True
            sel_sym = synonyms_sorted[i]
            break
                                  
      if changed:    
        
        for indx in word_idx_dic[str(target_word)]:
          #print("changed")
          text_prime[indx] = sel_sym
          text_prime = vowel_correction(text_prime[:] , indx)
          backtrack_dic[(sel_sym,indx)] = target_word
          num_changed += 1
  #print(num_changed)            
  text_prime = ' '.join(text_prime)
  probs = tmodel.getPredictions([text_prime])
  return text_prime, num_changed, orig_label, probs[0], num_queries


def main():
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--dataset_path",
                      type=str,
                      required=True,
                      help="Which dataset to attack.")

  parser.add_argument("--target_model",
                      type=str,
                      required=True,
                      choices=['wordLSTM', 'bert', 'wordCNN','gcp'],
                      help="Target models for text classification: wordcnn, word Lstm or GCP NLP API ")

  
  parser.add_argument("--gcp_nlp_json_link",
                      type=str,
                      required=False,
                      default='',
                      help="Link to GCP NLP API json key file")
  
  parser.add_argument("--target_model_path",
                      type = str,
                      required=False,
                      default='',
                      help="pre-trained target model path")

  parser.add_argument("--nclasses",
                      type = int,
                      default = 2,
                      help="How many classes for classification.")
  
  parser.add_argument("--word_embeddings_path",
                      type=str,
                      default='word_embeddings_path/glove.6B.200d.txt',
                      help="path to the word embeddings for the target model")
  parser.add_argument("--counter_fitting_embeddings_path",
                      type=str,
                      required=False,
                      default="counter_fitting_embedding/counter-fitted-vectors.txt",
                      help="path to the counter-fitting embeddings used to find synonyms")
  
  parser.add_argument("--output_dir",
                      type=str,
                      default='adv_results',
                      help="The output directory where the attack results will be written.")

  parser.add_argument("--pos_filter",
                      type=str,
                      default='coarse',
                      help="pos filter mask: either 'fine' or 'coarse")

  args = parser.parse_args()
  output_dir = args.output_dir

  #download data to be Attacked
  data = download_attack_data(args.dataset_path)
  data = data[:100]
  
  #find word2idx and idx2word dicts
  embeddings, word2idx_vocab, idx2word_vocab = generate_embedding_mat(args.counter_fitting_embeddings_path)

  #compute cosine similarity matrix of words in text
  cos_sim , word2idx_rev, idx2word_rev = generateCosSimMat.csim_matrix(data, embeddings, word2idx_vocab)

  gcp_attack = False
  #Load the saved model using state dic
  if args.target_model == "wordCNN":
    default_model_path = "saved_models/wordCNN/"
    if 'imdb' in args.dataset_path:
      default_model_path += 'imdb'
    elif 'mr' in args.dataset_path:
      default_model_path += 'mr'
        
    cmodel = Model(args.word_embeddings_path, nclasses = args.nclasses, hidden_size=100, cnn=True).cuda()
      
  elif args.target_model == "wordLSTM":
    default_model_path = "saved_models/wordLSTM/"
    if 'imdb' in args.dataset_path:
      default_model_path += 'imdb'
    elif 'mr' in args.dataset_path:
      default_model_path += 'mr'
    elif 'ag' in args.dataset_path:
      default_model_path += 'ag'
    cmodel=Model(args.word_embeddings_path, nclasses = args.nclasses, cnn = False).cuda()

  elif args.target_model == "bert":
    default_model_path = "saved_models/bert/"
    if 'imdb' in args.dataset_path:
      default_model_path += 'imdb'
    elif 'mr' in args.dataset_path:
      default_model_path += 'mr'
    if args.target_model_path: 
      cmodel = LoadPretrainedBert.loadPretrainedModel(args.target_model_path,nclasses = args.nclasses)
    else:
      cmodel = LoadPretrainedBert.loadPretrainedModel(default_model_path,nclasses = args.nclasses)

  elif args.target_model=="gcp":
    cmodel = None
      

  if args.target_model!='bert' and args.target_model!='gcp':
    #load checkpoints
    if args.target_model_path:
      print("target model path")          
      checkpoint = torch.load(args.target_model_path, map_location=torch.device('cuda:0'))
    else:
      checkpoint = torch.load(default_model_path, map_location=torch.device('cuda:0'))
        
    cmodel.load_state_dict(checkpoint)

  
  orig_failures = 0.
  adv_failures = 0.
  changed_rates = []
  nums_queries = []
  orig_texts = []
  adv_texts = []
  true_labels = []
  new_labels = []
  sem_sim = []
  log_file = open(os.path.join(args.output_dir,'results_log'), 'a')
  stop_words_set = criteria.get_stopwords()
  true_label = 1
  predictor = 1
  sim_score_threshold = 0.5
  perturb_ratio = 0.4
  size = len(data)
  print('Start attacking!')
  ct=0
  #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  pos_filter = args.pos_filter
  random_atk = False
  for idx, (text, true_label) in enumerate(data):
    ct+=1    
    print(idx)
    if idx % 20 == 0:
      print('{} samples out of {} have been finished!'.format(idx, size))

    new_text, num_changed, orig_label, \
    new_label, num_queries = attack(cmodel, args.gcp_nlp_json_link, text, true_label, stop_words_set,
                                              word2idx_rev, idx2word_rev, idx2word_vocab, cos_sim, pos_filter,
                                              synonym_num=80, sim_score_threshold=sim_score_threshold ,  
                                              syn_sim=0.65)

    #print(text)    
    if true_label != orig_label:
      orig_failures += 1
    else:
      nums_queries.append(num_queries)

    if true_label != new_label:
      adv_failures += 1

    tokenizer = RegexpTokenizer(r'\w+')
    text_tokens = tokenizer.tokenize(text)  
    changed_rate = 1.0 * num_changed / len(text_tokens)

    if true_label == orig_label and true_label != new_label:
      changed_rates.append(changed_rate)
      orig_texts.append(text)
      adv_texts.append(new_text)
      true_labels.append(true_label)
      new_labels.append(new_label)
        
  message = 'For target model {}: original accuracy: {:.3f}%, adv accuracy: {:.3f}%, ' \
                'avg changed rate: {:.3f}%, Avg num of queries: {:.1f}\n, Median num of queries:{:.1f} \n'.format(predictor,
                                                                        (1-orig_failures/size)*100,
                                                                              (1-adv_failures/size)*100,
                                                                        np.mean(changed_rates)*100,
                                                                        np.mean(nums_queries),
                          
                                                                        np.median(nums_queries))
  print(message)
  log_file.write(message)
  i=1
  with open(os.path.join(args.output_dir,'adversaries_AG_1-1000_lstm_coarse_POS.txt'), 'w') as ofile:
    for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
      ofile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))


    
if __name__ == "__main__":
    main()
    
