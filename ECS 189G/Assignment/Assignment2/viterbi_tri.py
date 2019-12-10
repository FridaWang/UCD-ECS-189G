import math, collections
import sys
import re

init_state = "init"
final_state = "final"
OOV_symbol = "OOV"

transition = collections.defaultdict(lambda: 0)
emission = collections.defaultdict(lambda: 0)
States = collections.defaultdict(lambda: 0)
Voc = collections.defaultdict(lambda: 0)

filename = sys.argv[1]
f = open(filename)

for line in f:
	
	words = line.split()
	if words[0] == "trans":
		preTag_1 = words[1]
		preTag_2 = words[2]
		curTag = words[3]
		probability = words[4]
		transition[preTag_1, preTag_2, curTag] = math.log(float(probability))
		States[preTag_1] = 1
		States[preTag_2] = 1
		States[curTag] = 1
	else:
		tag = words[1]
		Word = words[2]
		probability = words[3]
		emission[tag, Word] = math.log(float(probability))
		States[tag] = 1
		Voc[Word] = 1

f.close()

# read in one line each time
for line in sys.stdin:

	wordList = line.rstrip().split()
	n = len(wordList)
	wordList.insert(0, "")
	wordList.insert(0, "")

	Backtrace = collections.defaultdict(lambda: 0)
	Pi = collections.defaultdict(lambda: 0)

	Pi[0, init_state, init_state] = 0.0
	Pi[1, init_state, init_state] = 0.0

	for k in range(2, n+2):

		if wordList[k] not in Voc:
			wordList[k] = OOV_symbol

		for curTag in States.keys():
			for preTag_2 in States.keys():
				for preTag_1 in States.keys():
					if (preTag_1, preTag_2, curTag) in transition \
					and (curTag, wordList[k]) in emission \
					and (k-1, preTag_1, preTag_2) in Pi:
						pi = Pi[k-1, preTag_1, preTag_2] + \
						transition[preTag_1, preTag_2, curTag] + \
						emission[curTag, wordList[k]]
						if (k, preTag_2, curTag) not in Pi \
						or (pi > Pi[k, preTag_2, curTag]):
							Pi[k, preTag_2, curTag] = pi
							Backtrace[k, preTag_2, curTag] = preTag_1

	foundgoal = 0
	goal = 0
	pre1 = ""
	pre2 = ""
	for preTag_1 in States.keys():
		for preTag_2 in States.keys():
			if (preTag_1, preTag_2, final_state) in transition \
			and (n+1, preTag_1, preTag_2) in Pi:
				pi = Pi[n+1, preTag_1, preTag_2] + \
				transition[preTag_1, preTag_2, final_state]
				if foundgoal == 0 or pi > goal:
					goal = pi
					foundgoal = 1
					pre1 = preTag_1
					pre2 = preTag_2

	if foundgoal != 0 :
		t = []
		for i in range(n+1, 1, -1):
			t.append(pre2)
			temp = pre1
			pre1 = Backtrace[i, pre1, pre2]
			pre2 = temp
		t.reverse()
		print(" ".join(t))
	else: 
		print('')
	
		
