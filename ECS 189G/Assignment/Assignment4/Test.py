import getopt
import os
import math, collections
import operator
import sys
class NaiveBayes:
    class TrainSplit:
        """
        Set of training and testing data
        """
        def __init__(self):
            self.train = []
            self.test = []

    class Document:
        """
        This class represents a document with a label. classifier is 'pos' or 'neg' while words is a list of strings.
        """
        def __init__(self):
            self.classifier = ''
            self.words = []

    def __init__(self):
        """
        Initialization of naive bayes
        """
        self.stopList = set(self.readFile('data/english.stop'))
        self.bestModel = False
        self.stopWordsFilter = False
        self.naiveBayesBool = False
        self.numFolds = 10

        self.trained = False
        # number of documents in D
        self.N_doc = 0
        # number of documents from D in class c
        self.N_c = collections.defaultdict(lambda: 0)
        self.Voc = set()
       
        self.posCount = collections.defaultdict(lambda: 0)
        self.negCount = collections.defaultdict(lambda: 0)

        self.posDocFreq = collections.defaultdict(lambda: 0)
        self.negDocFreq = collections.defaultdict(lambda: 0)
        # log(P(c))
        self.logprior = collections.defaultdict(lambda: 0)
        # log(P(w|c))
        self.loglikehood = collections.defaultdict(lambda: 0)

        self.Alpha = 1

        self.negationWord = set({"never","neither","nor","not","don't","didn't","doesn't","isn't","aren't","wasn't","won't"})
        self.strongPos = set()       
        self.strongNeg = set()

        self.strongWeight = 0.3
        self.weakWeight = 0.15
        self.strongWeightedWords = set({"annoying","perfect","fantastic","groundbreaking","masterpiece","flawless","terrible","awful","bitch","fuck","ass","award","benefit","awesome","best"})
        self.weakWeightedWords = set({"like","likes","liked","blame","bored","boring","ashamed","annoyed","anti","anxious","anxiety","appreciation","approval","approved"})
        #self.strongNeg = set({"terrible","awful","bored","boring","bitch","fuck","ass","annoying"})
        self.punc = set({"." ,",", "?", "!", "-","\\","\"","(",":",")","\'"})
        
        # TODO
        # Implement a multinomial naive bayes classifier and a naive bayes classifier with boolean features. The flag
        # naiveBayesBool is used to signal to your methods that boolean naive bayes should be used instead of the usual
        # algorithm that is driven on feature counts. Remember the boolean naive bayes relies on the presence and
        # absence of features instead of feature counts.

        # When the best model flag is true, use your new features and or heuristics that are best performing on the
        # training and test set.

        # If any one of the flags filter stop words, boolean naive bayes and best model flags are high, the other two
        # should be off. If you want to include stop word removal or binarization in your best performing model, you
        # will need to write the code accordingly.

    def classify(self, words):
        """
        Classify a list of words and return a positive or negative sentiment
        """
       
        if self.stopWordsFilter:
            words = self.filterStopWords(words)

        # TODO
        # classify a list of words and return the 'pos' or 'neg' classification
        # Write code here

        self.logprior['pos'] = -math.log(float(self.N_c['pos']) / (self.N_c['pos'] + self.N_c['neg']))
        self.logprior['neg'] = -math.log(float(self.N_c['neg']) / (self.N_c['pos'] + self.N_c['neg']))

        if self.trained == False:
            if self.naiveBayesBool == True:
                self.calculateBinaryProbability() 
                self.trained = True
            elif self.bestModel == True:
                self.calculateBestModelProbability()
                self.trained = True
            else:
                self.calculateProbability()
                self.trained = True

        posProb = self.logprior['pos']
        negProb = self.logprior['neg']

        if self.naiveBayesBool == True or self.bestModel == True:
            words = set(words)

        for test_word in words:
            if test_word in self.Voc:
                posProb -= self.loglikehood[test_word,'pos']
                negProb -= self.loglikehood[test_word,'neg']
        
        

        if posProb < negProb:
            return 'pos'
        else:
            return 'neg'

    def calculateProbability(self):
        V = len(self.Voc)
        totalPosCount = sum(self.posCount.values())
        totalNegCount = sum(self.negCount.values())
        for word in self.Voc:
            self.loglikehood[word,'pos'] = math.log(float(self.posCount[word] + 1) / (totalPosCount + V))
            self.loglikehood[word,'neg'] = math.log(float(self.negCount[word] + 1) / (totalNegCount + V))

    def calculateBestModelProbability(self):
        V = len(self.Voc)
        totalPosDoc = sum(self.posDocFreq.values())
        totalNegDoc = sum(self.negDocFreq.values())

        negationFlag = False
        count = 0 
        for word in self.Voc:
            if word in self.punc and negationFlag == True:
                negationFlag = False
            if negationFlag == True:
                word = "NOT_" + word
                count += 1
                if count == 1:
                    negationFlag = False
                    count = 0
            if word in self.negationWord:
                negationFlag = True

            self.loglikehood[word,'pos'] = math.log(1.0 *(self.posDocFreq[word] + self.Alpha) / (totalPosDoc + 1 * V))
            self.loglikehood[word,'neg'] = math.log(1.0 *(self.negDocFreq[word] + self.Alpha) / (totalNegDoc + 1 * V))
    
    def calculateBinaryProbability(self):
        V = len(self.Voc)
        totalPosDoc = sum(self.posDocFreq.values())
        totalNegDoc = sum(self.negDocFreq.values())
        for word in self.Voc:
            self.loglikehood[word,'pos'] = math.log(1.0 *(self.posDocFreq[word] + self.Alpha) / (totalPosDoc + self.Alpha * V))
            self.loglikehood[word,'neg'] = math.log(1.0 *(self.negDocFreq[word] + self.Alpha) / (totalNegDoc + self.Alpha * V))
    

    def addDocument(self, classifier, words):
        """
        Train your model on a document with label classifier (pos or neg) and words (list of strings). You should
        store any structures for your classifier in the naive bayes class. This function will return nothing
        """
        # TODO
        # Train model on document with label classifiers and words
        # Write code here
        presence = set()
        self.N_doc += 1
        self.N_c[classifier] += 1

        negation = False
        count = 0
        for word in words:

            if self.bestModel == True:
                if word in self.punc and negation == True:
                    negation = False
                if negation == True:
                    word = "NOT_" + word
                    count += 1
                    if count == 1:
                        negation = False
                        count = 0
                if word in self.negationWord:
                    negation = True

            presence.add(word)

            if classifier == 'pos':
                self.posCount[word] += 1
            else:
                self.negCount[word] += 1
            if word not in self.Voc:
                self.Voc.add(word)

        for word in presence:
            if classifier == 'pos':
                self.posDocFreq[word] += 1
                if self.bestModel:
                    if word in self.strongWeightedWords:
                        self.posDocFreq[word] += self.strongWeight
                    if word in self.weakWeightedWords:
                        self.posDocFreq[word] += self.strongWeight
            else:
                self.negDocFreq[word] += 1
                if self.bestModel:
                    if word in self.strongWeightedWords:
                        self.negDocFreq[word] += self.strongWeight
                    if word in self.weakWeightedWords:
                        self.negDocFreq[word] += self.strongWeight
                        
        
    def readFile(self, fileName):
        """
        Reads a file and segments.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        str = '\n'.join(contents)
        result = str.split()
        return result

    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posDocTrain = os.listdir('%s/pos/' % trainDir)
        negDocTrain = os.listdir('%s/neg/' % trainDir)
        for fileName in posDocTrain:
            doc = self.Document()
            doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            doc.classifier = 'pos'
            split.train.append(doc)
        for fileName in negDocTrain:
            doc = self.Document()
            doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            doc.classifier = 'neg'
            split.train.append(doc)
        return split

    def train(self, split):
        for doc in split.train:
            words = doc.words
            if self.stopWordsFilter:
                words = self.filterStopWords(words)
            self.addDocument(doc.classifier, words)

    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posDocTrain = os.listdir('%s/pos/' % trainDir)
        negDocTrain = os.listdir('%s/neg/' % trainDir)
        # for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                doc.classifier = 'pos'
                if fileName[2] == str(fold):
                    split.test.append(doc)
                else:
                    split.train.append(doc)
            for fileName in negDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                doc.classifier = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(doc)
                else:
                    split.train.append(doc)
            yield split

    def test(self, split):
        """Returns a list of labels for split.test."""
        labels = []
        for doc in split.test:
            words = doc.words
            if self.stopWordsFilter:
                words = self.filterStopWords(words)
            guess = self.classify(words)
            labels.append(guess)
        return labels

    def buildSplits(self, args):
        """
        Construct the training/test split
        """
        splits = []
        trainDir = args[0]
        if len(args) == 1:
            print '[INFO]\tOn %d-fold of CV with \t%s' % (self.numFolds, trainDir)

            posDocTrain = os.listdir('%s/pos/' % trainDir)
            negDocTrain = os.listdir('%s/neg/' % trainDir)
            for fold in range(0, self.numFolds):
                split = self.TrainSplit()
                for fileName in posDocTrain:
                    doc = self.Document()
                    doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                    doc.classifier = 'pos'
                    if fileName[2] == str(fold):
                        split.test.append(doc)
                    else:
                        split.train.append(doc)
                for fileName in negDocTrain:
                    doc = self.Document()
                    doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                    doc.classifier = 'neg'
                    if fileName[2] == str(fold):
                        split.test.append(doc)
                    else:
                        split.train.append(doc)
                splits.append(split)
        elif len(args) == 2:
            split = self.TrainSplit()
            testDir = args[1]
            print '[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir)
            posDocTrain = os.listdir('%s/pos/' % trainDir)
            negDocTrain = os.listdir('%s/neg/' % trainDir)
            for fileName in posDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                doc.classifier = 'pos'
                split.train.append(doc)
            for fileName in negDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                doc.classifier = 'neg'
                split.train.append(doc)

            posDocTest = os.listdir('%s/pos/' % testDir)
            negDocTest = os.listdir('%s/neg/' % testDir)
            for fileName in posDocTest:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (testDir, fileName))
                doc.classifier = 'pos'
                split.test.append(doc)
            for fileName in negDocTest:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (testDir, fileName))
                doc.classifier = 'neg'
                split.test.append(doc)
            splits.append(split)
        return splits

    def filterStopWords(self, words):
        """
        Stop word filter
        """
        removed = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                removed.append(word)
        return removed


def test10Fold(args, stopWordsFilter, naiveBayesBool, bestModel):
    nb = NaiveBayes()
    splits = nb.buildSplits(args)
    avgAccuracy = 0.0
    fold = 0
    for split in splits:
        classifier = NaiveBayes()
        classifier.stopWordsFilter = stopWordsFilter
        classifier.naiveBayesBool = naiveBayesBool
        classifier.bestModel = bestModel
        accuracy = 0.0
        for doc in split.train:
            words = doc.words
            classifier.addDocument(doc.classifier, words)

        #count = 0
        for doc in split.test:
            words = doc.words
            guess = classifier.classify(words)
            if doc.classifier == guess:
                accuracy += 1.0

        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print '[INFO]\tAccuracy: %f' % avgAccuracy


def classifyFile(stopWordsFilter, naiveBayesBool, bestModel, trainDir, testFilePath):
    classifier = NaiveBayes()
    classifier.stopWordsFilter = stopWordsFilter
    classifier.naiveBayesBool = naiveBayesBool
    classifier.bestModel = bestModel
    trainSplit = classifier.trainSplit(trainDir)
    classifier.train(trainSplit)
    testFile = classifier.readFile(testFilePath)
    print classifier.classify(testFile)


def main():
    stopWordsFilter = False
    naiveBayesBool = False
    bestModel = False
    (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
    if ('-f', '') in options:
        stopWordsFilter = True
    elif ('-b', '') in options:
        naiveBayesBool = True
    elif ('-m', '') in options:
        bestModel = True

    if len(args) == 2 and os.path.isfile(args[1]):
        classifyFile(stopWordsFilter, naiveBayesBool, bestModel, args[0], args[1])
    else:
        test10Fold(args, stopWordsFilter, naiveBayesBool, bestModel)


if __name__ == "__main__":
    main()
