import scipy
import tensorflow_hub as hub
#import tensorflow as tf
from scipy import spatial
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)

def embed(input):
      #print(type(input))
      return model(input)

def semantic_sim(a,b):
    u=embed(a).numpy()
    v=embed(b).numpy()
    dist=spatial.distance.cosine(u, v)
    return 1-dist
