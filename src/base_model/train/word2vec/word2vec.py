import tensorflow as tf
from keras import Model
from keras.layers import  Dot, Embedding, Flatten, Reshape



""" 
*********************************
Word2Vec class
*********************************** 
"""

class Word2Vec(Model):
    def __init__(self, vocab_size=2000, embedding_dim=512, num_ns=4):
      super(Word2Vec, self).__init__()
      self.target_embedding = Embedding(vocab_size, 
                                        embedding_dim,
                                        input_length=1,
                                        name="w2v_embedding", )
      self.context_embedding = Embedding(vocab_size, 
                                        embedding_dim, 
                                        input_length=num_ns+1,
                                        name = "ctxt_embedding")
      self.dots = Dot(axes=(3,1))
      self.flatten = Flatten()

    def call(self, pair):
      target, context = pair
      we = self.target_embedding(target)
      ce = self.context_embedding(context)
      dots = self.dots([ce, we])
      return self.flatten(dots)
