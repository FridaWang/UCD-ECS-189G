import math, collections

class BackoffModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    # Tip: To get words from the corpus, try
    #    for sentence in corpus.corpus:
    #       for datum in sentence.data:  
    #         word = datum.word
    for sentence in corpus.corpus:
        self.unigramCounts["<s>"] +=1
        self.total += 1
        for i in xrange(1, len(sentence.data)):  
            word_pre = sentence.data[i-1].word
            word_cur = sentence.data[i].word
            self.unigramCounts[word_cur] = self.unigramCounts[word_cur] + 1
            self.bigramCounts[word_cur, word_pre] = self.bigramCounts[word_cur, word_pre] + 1
            self.total += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0 
    for i in xrange(1,len(sentence)):
        word_pre = sentence[i-1]
        word_cur = sentence[i]  
        countBigram = self.bigramCounts[word_cur, word_pre]
        countUnigram = self.unigramCounts[word_pre]
        if countBigram > 0:
            score += math.log(countBigram)
            score -= math.log(countUnigram)
        else:
            score += math.log(self.unigramCounts[word_cur] + 1)
            score -= math.log(self.total + len(self.unigramCounts)) 
    return score
    
