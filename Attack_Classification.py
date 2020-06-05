import nltk
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pattern
import criteria
import torch
import dataloader
import argparse
import os
import numpy as np
import dataloader
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
from lookup_dicts import generate_lookup_dicts

nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')


class model:
    def __init__(self,cmodel):
        self.cmodel=cmodel
        
    # prediction function for classsification task
    def getPredictions(self,revs):
        '''
            revs: List of texts
        '''
        revs1=[rev.split(" ") for rev in revs]
        orig_probs=self.cmodel.text_pred(revs1).squeeze()
        
        if len(revs)>1:
            orig_label = torch.argmax(orig_probs,axis=1)
        else:
            orig_label = torch.argmax(orig_probs)
            return [orig_label.tolist()]
        
        return orig_label.tolist()


#Attack function
def attack(text_ls, true_label, cmodel, stop_words_set, word2idx, idx2word, cos_sim, sim_score_threshold=0.5, 
           sim_score_window=15, synonym_num=50,syn_sim=0.75,):
    """
        Attacks text 

        Arguments:
        text_ls: str, the text to be attacked
        true_label: int, representing true class of text_ls
        cmodel: Model to be attacked
        cos_sim: numpy array, precomuted cosine similarity square matrix
        word2idx: dict mapping words to indices in the precomuted cosine similarity square matrix
        idx2word: dict mapping indices of precomuted cosine similarity square matrix back to words
        sim_score_threshold: float,semantic similarity threshold while selecting or rejecting synonyms,default:0.5
        sim_score_window: int,window size for computing semantic similarity between actual and perturbed text around the perturbed word
        synonym_num: int,max number of candidate synonyms to be analysed
        syn_sim: float, threshold for cosine similarity between candidate synonyms and original word,defualt:0.75 
 
    """
    tmodel=model(cmodel)
    
    orig_text1=text_ls[:]
    orig_label=tmodel.getPredictions([text_ls])[0]
    labels=['Negative','Positive']
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        text_temp=text_ls
        nlp = spacy.load('en')
        doc=nlp(str(text_ls))
        text_ls=[str(j) for j in doc]
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1
        
        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)
        
        #sentence segmentation
        sents_sentiment_dic={}
        text_sentences = nlp(text_temp)
        sents1=text_sentences.sents
        sents=[str(sent) for sent in sents1]
        
        #segregate positive and negative sentence
        preds=tmodel.getPredictions(list(sents))
        num_queries+=len(sents)
        sents_sentiment_dic[orig_label]=[]
        sents_sentiment_dic[1-orig_label]=[]
        for i in range(len(preds)):
            if preds[i] in sents_sentiment_dic:
                sents_sentiment_dic[preds[i]].append(sents[i])
            else:
                sents_sentiment_dic[preds[i]]=[sents[i]]
                
        #get sentence importance ranking
        try:
            pos_aggregate=' '.join(sents_sentiment_dic[1-orig_label])
        except:
            pos_aggregate=""
            
        neg_sents=sents_sentiment_dic[orig_label].copy()

        p=0
        top_sent_imp={}
        agg_imp_dic={}
        word_agg_dic1={}
        sent_agg_dic={}
        while len(neg_sents)!=0:
            p+=1
            top_sent_imp[p]=[]
            if p<=len(neg_sents):
                for neg_sent_comb in combinations(neg_sents,p):
                    neg_agg=' '.join(list(neg_sent_comb))
                    agg_=pos_aggregate + " " + neg_agg
                    
                    preds3=tmodel.getPredictions([agg_])[0]
                    num_queries+=1
                    if preds3==orig_label:
                            top_sent_imp[p].extend(list(neg_sent_comb))
                            conv_set=set(top_sent_imp[p])
                            top_sent_imp[p]=list(conv_set)
                            agg_imp_dic[agg_]=1-orig_label
                            tokenizer = RegexpTokenizer(r'\w+')
                            text_tokens = tokenizer.tokenize(agg_)
                        
                            for word in text_tokens:
                                if len(word)>1:
                                    #print(word)
                                    if word in word_agg_dic1:
                                        word_agg_dic1[word].append(agg_)
                                    else:
                                        word_agg_dic1[word]=[agg_]

                
                    if len(top_sent_imp[p])!=0:
                            for sent4 in top_sent_imp[p]:
                                   if sent4 in neg_sents:
                                        del neg_sents[neg_sents.index(sent4)]
            else:
                         top_sent_imp[p].extend(list(neg_sents))
                         neg_sents=[]
                         break

        #create sent2imp dictionary
        imp_dic={}
        for key in top_sent_imp:
                for sent6 in top_sent_imp[key]:
                            imp_dic[sent6]=key
         
        #get word importance scores
        ind_count=0
        import_scores=[]
        for sent in sents:
            text_tokens = nlp(sent)
            text_sent = [str(word) for word in text_tokens]
            
            if not sent in sents_sentiment_dic[orig_label]:
                    import_scores.extend([300]*len(text_sent))
            else:                                 
                    pos_tags=criteria.get_pos(text_sent)
                    for i1 in range(len(text_sent)):
                        if pos_tags[i1]=='ADV':
                                import_scores.append(imp_dic[sent]+15)
                        elif pos_tags[i1]=='VERB':
                                import_scores.append(imp_dic[sent]+15)
                        elif pos_tags[i1]=='ADJ':
                                import_scores.append(imp_dic[sent])
                        else:
                            import_scores.append(imp_dic[sent]+50)
                        if len(text_sent[i1])>1 and 1<0:
                            if text_sent[i1] in word_agg_dic1 and not sent_agg_dic.get(sent,[sent]) in word_agg_dic1[text_sent[i1]]:
                                    word_agg_dic1[text_sent[i1]].append(sent_agg_dic.get(sent,[sent]))
                            else:
                                    word_agg_dic1[text_sent[i1]]=[sent_agg_dic.get(sent,[sent])]
                        

        import_scores=np.array(import_scores)
        
        # get words to perturb ranked by importance score for word in words_perturb
        words_perturb = []
        text_prime = text_ls[:]
        import_scores=np.array(import_scores)
        imp_indxs=np.argsort(import_scores).tolist()

        for idx in imp_indxs:
            try:
                if not text_prime[idx] in stop_words_set:
                    words_perturb.append((idx, text_prime[idx]))
            except:
                continue
            
        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, syn_sim)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        num_changed = 0
        idx_flag=0
        backtrack_dic={}
        flg=0
        misclassified=False
        visited={}
        
        #map the words to list indices at which it is present in original (text_prime)
        word_idx_dic={}
        for idx in range(len(text_prime)):
            word=text_prime[idx]
            if word in word_idx_dic:
                word_idx_dic[word].append(idx)
            else:
                word_idx_dic[word]=[idx]
            visited[word]=False
        origtext_prime=text_prime.copy()

        for idx, synonyms in synonyms_all:
                orig_pos=criteria.get_pos(text_prime)[idx]
                idx=idx+idx_flag
                if len(origtext_prime[idx])<=1 or visited[origtext_prime[idx]]==True:
                    continue

                if misclassified:   #indicator of misclassification 
                    #backtrack: revert those replacements which were not necessary to make the adversary minimally perturbed

                    for (wrd,index) in backtrack_dic:

                        pred=tmodel.getPredictions([' '.join(text_prime[:index]+[backtrack_dic[(wrd,index)]]+text_prime[index+1:])])[0]
                        num_queries+=1
                        if pred==(1-orig_label):
                            text_prime[index]=backtrack_dic[(wrd,index)]
                            num_changed-=1
                    break
    

                # Step#1: Find all aggregates(with orig_label) to which the target wrd belongs
                target_word=text_prime[idx]
                
                visited[target_word]=True
                flg3=0
                if target_word in word_agg_dic1:
                    if word_agg_dic1[target_word]==[]:
                        word_agg_dic1.pop(target_word)
                    else:
                        agg_list=list(set(word_agg_dic1[target_word]))
                        word_agg_dic1[target_word]=[]

                        flg3=1
                    
                if flg3==0:
                    for sent in sents_sentiment_dic[orig_label]:
                        if target_word in sent:
                            agg_list=[sent]
                            flg3=1
                            break
                    if flg3==0:
                        continue

                        
                #Replace the word with all the synonyms
                agg_with_synonyms=[[agg.replace(target_word,synonym) for synonym in synonyms] for agg in agg_list]
                
                #Query the model for reviews after replacements with the synonyms for the word 
                for agg_with_syn in agg_with_synonyms:
                    preds4=tmodel.getPredictions(agg_with_syn)

                    num_queries+=len(preds4)
 
                    orig_pos=criteria.get_pos(text_prime)[idx]

                    ind=-1
                    
                    #Find the synonym replacing which changes the aggregate to misclassify
                    if (1-orig_label) in preds4:
                        flg=0
                        ct1=preds4.count(int(1-orig_label))
                        flag2=0
                        bestsim=-1
                        while flag2<ct1:
            
                                    ind=preds4.index(int(1-orig_label),ind+1)
                                    flag2+=1
                                    sym_wrd=synonyms[ind]
                                    prd1=tmodel.getPredictions([' '.join(text_prime[:idx]+[sym_wrd]+text_prime[idx+1:])])[0]
                                    new_rev=text_prime.copy()
                                    new_rev[idx]=sym_wrd
                                    orig_word=text_prime[idx]
                                    orig_text_prime=' '.join(text_prime)
                                    sem_sim=semantic_sim([' '.join(new_rev)],[orig_text_prime])
                                    pos_tag_new=criteria.get_pos(new_rev)[idx]

                                    if sem_sim>=sim_score_threshold and orig_pos==pos_tag_new:
                                            if prd1==(1-orig_label):
                                                    sel_word=sym_wrd
                                                    flg=1
                                                    misclassified=True
                                                    break

                                            elif sem_sim>bestsim:
                                                sel_word=sym_wrd
                                                bestsim=sem_sim
                                                flg=1

                                
                        if flg==1:
                            #change the target_word with sel_word at all the places in the review
                            if not misclassified:
                                for indx in word_idx_dic[str(target_word)]:
                                    text_prime[indx]=sel_word
                                    backtrack_dic[(sel_word,indx)]=target_word
                                    num_changed+=1
                            else:
                                text_prime[idx]=sel_word
                                backtrack_dic[(sel_word,idx)]=target_word
                                num_changed+=1
                            break

                    continue
                    
        
        text_prime=' '.join(text_prime)
        probs=tmodel.getPredictions([text_prime])
        
        return text_prime, num_changed, orig_label,probs[0], num_queries


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
                        choices=['wordLSTM', 'bert', 'wordCNN'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm ")
    parser.add_argument("--target_model_path",
                        type=str,
                        required=False,
                        default='',
                        help="pre-trained target model path")
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='word_embeddings_path/glove.6B.200d.txt',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        required=False,
                        default="counter_fitting_embedding/counter-fitted-vectors.txt",
                        help="path to the counter-fitting embeddings used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='',
                        required=False,
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    
    parser.add_argument("--output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")
    
    args = parser.parse_args()
    output_dir=args.output_dir

    #download data to be Attacked
    data=download_attack_data(args.dataset_path)

    #find word2idx and idx2word dicts
    word2idx,idx2word=generate_lookup_dicts(args.counter_fitting_embeddings_path)

    #Load the saved model using state dic
    if args.target_model=="wordCNN":
        default_model_path="saved_models/wordCNN/"
        if 'imdb' in args.dataset_path:
            default_model_path+='imdb'
        elif 'yelp' in args.dataset_path:
            default_model_path+='yelp'
            
        cmodel = Model(args.word_embeddings_path, nclasses=2, hidden_size=100, cnn=True)
    elif args.target_model=="wordLSTM":
        default_model_path="saved_models/wordLSTM/"
        if 'imdb' in args.dataset_path:
            default_model_path+='imdb'
        elif 'yelp' in args.dataset_path:
            default_model_path+='yelp'
            
        cmodel=Model(args.word_embeddings_path, nclasses=2, hidden_size=100, cnn=False)

    #load checkpoints
    if args.target_model_path:
        checkpoint = torch.load(args.target_model_path,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(default_model_path,map_location=torch.device('cpu'))
        
    cmodel.load_state_dict(checkpoint)

    #compute cosine similarity matrix between counter fitted words
    cos_sim=compute_cos_sim_mat(args.counter_fitting_embeddings_path,args.counter_fitting_cos_sim_path)
    
    data=data[:10]
    orig_failures = 0.
    adv_failures = 0.
    changed_rates = []
    nums_queries = []
    orig_texts = []
    adv_texts = []
    true_labels = []
    new_labels = []
    sem_sim=[]
    log_file = open(os.path.join(args.output_dir,'results_log'), 'a')
    stop_words_set = criteria.get_stopwords()
    true_label=1
    predictor=1
    sim_score_threshold=0.8
    perturb_ratio=0.4
    size=len(data)
    print('Start attacking!')
    ct=0
    #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    for idx, (text, true_label) in enumerate(data):
            ct+=1
            print(ct)
            if idx % 20 == 0:
                print('{} samples out of {} have been finished!'.format(idx, size))

            new_text, num_changed, orig_label, \
            new_label, num_queries= attack(text, true_label,cmodel, stop_words_set,
                                                word2idx, idx2word, cos_sim,synonym_num=80,
                                                sim_score_threshold=sim_score_threshold ,syn_sim=0.6)

            flag1=0
            if true_label != orig_label:
                orig_failures += 1
            else:
                nums_queries.append(num_queries)
            if true_label != new_label:
                adv_failures += 1
   
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(text)

            changed_rate = 1.0 * num_changed / len(tokens)

            if true_label == orig_label and true_label != new_label:
                changed_rates.append(changed_rate)
                orig_texts.append(text)
                adv_texts.append(new_text)
                true_labels.append(true_label)
                new_labels.append(new_label)

    message = 'For target model {}: original accuracy: {:.3f}%, adv accuracy: {:.3f}%, ' \
                  'avg changed rate: {:.3f}%, Avg num of queries: {:.1f}\n, Median num of queries:{:.1f} avg Semantic Similarity: {:.3f}'.format(predictor,
                                                                         (1-orig_failures/size)*100,
                                                                               (1-adv_failures/size)*100,
                                                                         np.mean(changed_rates)*100,
                                                                         np.mean(nums_queries),
                                                                        np.median(nums_queries))
    print(message)
    log_file.write(message)

    with open(os.path.join(args.output_dir,'adversaries.txt'), 'w') as ofile:
            for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
                ofile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))


    
if __name__ == "__main__":
    main()
    
