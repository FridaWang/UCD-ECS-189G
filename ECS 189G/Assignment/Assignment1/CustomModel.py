import math, collections
class CustomModel:

  def __init__(self, corpus):
    """Initial custom language model and structures needed by this mode"""
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.p_continuation = collections.defaultdict(lambda: 0)
    self.continuationCounts = collections.defaultdict(lambda: 0)
    self.nextWordTypes = collections.defaultdict(lambda: 0)
    self.d = 0.75
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
    """  
    # TODO your code here
    for sentence in corpus.corpus:
        self.unigramCounts["<s>"] +=1
        for i in xrange(1, len(sentence.data)):  
            word_pre = sentence.data[i-1].word
            word_cur = sentence.data[i].word
            if self.bigramCounts[word_pre, word_cur] == 0:
              self.continuationCounts[word_cur] += 1 
              self.nextWordTypes[word_pre] += 1 
            self.bigramCounts[word_pre, word_cur] += 1
            self.unigramCounts[word_cur] += 1


  def score(self, sentence):
    """ With list of strings, return the log-probability of the sentence with language model. Use
        information generated from train.
    """
    # TODO your code here
    score = 0.0 

    for i in xrange(1,len(sentence)):
        word_pre = sentence[i-1]
        word_cur = sentence[i]  
        self.p_continuation[word_cur] = 1.0 * self.continuationCounts[word_cur] / len(self.bigramCounts) 
        if self.unigramCounts[word_pre] == 0:
          p_kn = 1.0 * self.d / len(self.unigramCounts)
        else:
          lambda_pre =  1.0 * (self.d / self.unigramCounts[word_pre]) * self.nextWordTypes[word_pre]
          p_kn = max(self.bigramCounts[word_pre, word_cur] - self.d, 0) / self.unigramCounts[word_pre] + lambda_pre * self.p_continuation[word_cur]
        score += math.log(p_kn + 1e-12)
    return score
