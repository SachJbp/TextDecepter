# Hard-Label-Black-Box-Attack-on-NLP
Hard Label Black Box attack on NLP

Follow the steps to run the attack algorithm

1) Download the [counter-fitted-vectors.txt](https://drive.google.com/open?id=1JXznRuK-tfewW_KyNMuTElSa0JxXCkCx) and put it in counter_fitting_embedding folder

2) Download [glove embeddings](http://nlp.stanford.edu/data/glove.6B.zip), extract 'glove.6B.200d.txt' and put it in 'word_embeddings_path' folder

3) Download pretrained target state dicts from [CNN](https://drive.google.com/file/d/1yUHFGN0e8Q8v_NU5wW25wx27bEOAyL0P/view) ,[LSTM](https://drive.google.com/file/d/1jOcUzWj3lpmiXHVi_KzvDK_sWmsmx7B5/view) and put it under subdirectories 'wordCNN' & 'wordLSTM' in 'saved_models' folder.

4) Use the following syntax to run the attack algorithm

>!python Attack_Classification.py --dataset_path "data/imdb.txt" --target_model 'wordCNN'

dataset_path can be either "data/imdb.txt" or "data/yelp.txt" 

target_model can be either wordCNN or wordLSTM




