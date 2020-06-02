def generate_lookup_dicts(counter_fitting_embeddings_path):

        #Generate word2idx and idx2word dicts for quick lookup of words from indices of cosine similarity matrix and vice versa
        print("Building vocab")
        idx2word={}
        word2idx={}
        i=0
        with open(counter_fitting_embeddings_path, 'r') as ifile:
            try:    
                for line in ifile:
                    word = line.split()[0]
                    if word not in idx2word:
                        idx2word[len(idx2word)] = word
                        word2idx[word] = len(idx2word) - 1
            except:
                print("error")
                print(i)
                i+=1
        return word2idx,idx2word
