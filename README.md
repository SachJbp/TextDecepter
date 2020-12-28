# Hard-Label-Black-Box-Attack-on-NLP
TextDecepter: Hard Label Black Box attack on NLP

Note: The pretrained target models used for testing the attack algorithm have been taken from [Textfooler](https://github.com/jind11/TextFooler)

Follow the steps to run the attack algorithm

1) Download the [counter-fitted-vectors.txt](https://drive.google.com/open?id=1JXznRuK-tfewW_KyNMuTElSa0JxXCkCx) and put it in counter_fitting_embedding folder

2) Download [glove embeddings](http://nlp.stanford.edu/data/glove.6B.zip), extract 'glove.6B.200d.txt' and put it in 'word_embeddings_path' folder

3) Download pretrained target model parameters from [CNN](https://drive.google.com/file/d/1yUHFGN0e8Q8v_NU5wW25wx27bEOAyL0P/view) ,[LSTM](https://drive.google.com/file/d/1jOcUzWj3lpmiXHVi_KzvDK_sWmsmx7B5/view), [BERT](https://drive.google.com/drive/folders/1wKjelHFcqsT3GgA7LzWmoaAHcUkP4c7B?usp=sharing) and put it under subdirectories 'wordCNN', 'wordLSTM' and 'BERT' in 'saved_models' folder.

4) Use the following syntax to run the attack algorithm

>!python Attack_Classification.py --dataset_path 'data/imdb.txt' --target_model 'bert' --counter_fitting_embeddings_path "counter_fitting_embedding/counter-fitted-vectors.txt" --target_model_path "saved_models/bert/imdb" --word_embeddings_path "word_embeddings_path/glove.6B.200d.txt" --output_dir "adv_results" --pos_filter "coarse"

dataset_path can be either "data/imdb.txt" or "data/mr.txt" 

target_model can be either wordCNN , wordLSTM, bert or gcp

The result files can be accessed from [Google Drive link](https://drive.google.com/drive/folders/10QbZ10zFiyxP-Z8AwbckalWNew1x54cG?usp=sharing)
