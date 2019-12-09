import math, collections
import sys
import re

init_state = "init"
final_state = "final"
OOV_symbol = "OOV"

verbose = 0

transition = collections.defaultdict(lambda: 0)
emission = collections.defaultdict(lambda: 0)
States = collections.defaultdict(lambda: 0)
Voc = collections.defaultdict(lambda: 0)


filename = sys.argv[1]

f = open(filename)

for line in f:
	words = line.split()
	if words[0] == "trans":
		qq = words[1]
		q = words[2]
		p = words[3]
		transition[qq, q] = math.log(float(p))
		States[qq] = 1
		States[q] = 1
	else:
		q = words[1]
		w = words[2]
		p = words[3]
		emission[q, w] = math.log(float(p))
		States[q] = 1
		Voc[w] = 1

f.close()

# read in one line each time
for line in sys.stdin:
	w = line.split()
	n = len(w)
	w.insert(0, "")

	Backtrace = collections.defaultdict(lambda: 0)
	V = collections.defaultdict(lambda: 0)
	V[0, init_state] = 0.0
	for i in range(1, n+1):
		if w[i] not in Voc:
			w[i] = OOV_symbol
		for q in States.keys():
			for qq in States.keys():
				if (qq, q) in transition and (q,w[i]) in emission and (i-1, qq) in V:
					v = V[i-1, qq] + transition[qq, q] + emission[q, w[i]]
					if ((i, q) not in V) or (v > V[i, q]):
						V[i, q] = v
						Backtrace[i, q] = qq
						#print("bk=["+str(i)+","+str(q)+"]="+str(qq))

	foundgoal = 0
	goal = 0
	for qq in States.keys():
		if (qq, final_state) in transition and (n, qq) in V:
			v = V[n, qq] + transition[qq, final_state]
			if foundgoal == 0 or v > goal:
				goal = v
				foundgoal = 1
				q = qq


	if foundgoal != 0 :
		t = []
		for i in range(n, 0, -1):
			t.append(q)
			q = Backtrace[i, q]
			
		t.reverse()
		print(" ".join(t))
	else: 
		print('')
	
		
