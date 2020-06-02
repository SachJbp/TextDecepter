import numpy as np
print("Building cos sim matrix...")

def compute_cos_sim_mat(counter_fitting_embedding_path,counter_fit_cos_sim_path):
    print(counter_fit_cos_sim_path," sachin")
    if counter_fit_cos_sim_path!='':
            # load pre-computed cosine similarity matrix if provided
            print('Load pre-computed cosine similarity matrix from {}'.format(counter_fit_cos_sim_path))
            cos_sim = np.load(counter_fit_cos_sim_path)
    else:
            # calculate the cosine similarity matrix
            print('Start computing the cosine similarity matrix!')
            embeddings = []
        
            with open(counter_fitting_embedding_path, 'r') as ifile:
                    for line in ifile:
                        embedding = [float(num) for num in line.strip().split()[1:]]
                        embeddings.append(np.array(embedding))   
                    
            
            embeddings=np.array(embeddings)
            print(type(embeddings))
            print(embeddings.shape)
            product = np.dot(embeddings, embeddings.T)
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            cos_sim = product / np.dot(norm, norm.T)

    return cos_sim
    print("Cos sim import finished!")
