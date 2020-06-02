from nltk.tokenize import RegexpTokenizer


def download_attack_data(filepath):
    file=open(filepath)
    line=file.readline()
    labels=[]
    texts=[]
    while line:
        label=int(line[0])
        txt=line[2:]
            
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(txt)
        if len(list(tokens))<350:
            labels.append(int(line[0]))
            texts.append(line[2:])
        
        
        line=file.readline()
    data = list(zip(texts, labels))
    print("Data import finished!")
    return data
