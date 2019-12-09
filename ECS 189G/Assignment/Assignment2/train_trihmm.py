#!/usr/bin/python

# David Bamman
# 2/14/14
#
# Python port of train_hmm.pl:

# Noah A. Smith
# 2/21/08
# Code for maximum likelihood estimation of a bigram HMM from 
# column-formatted training data.

# Usage:  train_hmm.py tags text > hmm-file

# The training data should consist of one line per sequence, with
# states or symbols separated by whitespace and no trailing whitespace.
# The initial and final states should not be mentioned; they are 
# implied.  
# The output format is the HMM file format as described in viterbi.pl.

import sys,re
from itertools import izip
from collections import defaultdict

TAG_FILE=sys.argv[1]
TOKEN_FILE=sys.argv[2]

vocab={}
OOV_WORD="OOV"
INIT_STATE="init"
FINAL_STATE="final"

lambda_1 = 0.0
lambda_2 = 0.0
lambda_3 = 0.0

emissions = {}
transitions_tri = {}
transitionsTotal_tri = defaultdict(int)
transition_bi = {}
transitionsTotal_bi = defaultdict(int)
transition_uni = defaultdict(int)

emissionsTotal = defaultdict(int)

with open(TAG_FILE) as tagFile, open(TOKEN_FILE) as tokenFile:
	for tagString, tokenString in izip(tagFile, tokenFile):

		tags=re.split("\s+", tagString.rstrip())
		tokens=re.split("\s+", tokenString.rstrip())
		pairs=zip(tags, tokens)

		prevtag1 = INIT_STATE 
		prevtag2 = INIT_STATE

		transition_uni[INIT_STATE] += 1

		for (tag, token) in pairs:

			# this block is a little trick to help with out-of-vocabulary (OOV)
			# words.  the first time we see *any* word token, we pretend it
			# is an OOV.  this lets our model decide the rate at which new
			# words of each POS-type should be expected (e.g., high for nouns,
			# low for determiners).

			if token not in vocab:
				vocab[token] = 1
				token = OOV_WORD

			if tag not in emissions:
				emissions[tag] = defaultdict(int)

			if (prevtag1, prevtag2) not in transitions_tri:
				transitions_tri[prevtag1, prevtag2] = defaultdict(int)

			if prevtag2 not in transition_bi:
				transition_bi[prevtag2] = defaultdict(int)
			
			
			# increment the emission/transition observation
			emissions[tag][token] += 1
			emissionsTotal[tag] += 1
			
			transitions_tri[prevtag1, prevtag2][tag] += 1
			transitionsTotal_tri[prevtag1, prevtag2] += 1

			transition_bi[prevtag2][tag] += 1
			transitionsTotal_bi[prevtag2] += 1

			transition_uni[tag] += 1

			prevtag1 = prevtag2
			prevtag2 = tag

		# don't forget the stop probability for each sentence
		if (prevtag1, prevtag2) not in transitions_tri:
			transitions_tri[prevtag1, prevtag2] = defaultdict(int)

		if prevtag2 not  in transitionsTotal_bi:
			transition_bi[prevtag2] = defaultdict(int)


		transitions_tri[prevtag1, prevtag2][FINAL_STATE] += 1
		transitionsTotal_tri[prevtag1, prevtag2] += 1

		transition_bi[prevtag2][FINAL_STATE] += 1
		transitionsTotal_bi[prevtag2] += 1

		transition_uni[FINAL_STATE] += 1;

tagTotal = 0
for tag in transition_uni:
	tagTotal += transition_uni[tag]
# optimize lambada

for (prevtag1, prevtag2) in transitions_tri:
	for tag in transitions_tri[prevtag1, prevtag2]:
		if (transitionsTotal_tri[prevtag1, prevtag2] -1) != 0:
			num_tri = 1.0 * (transitions_tri[prevtag1, prevtag2][tag] -1) / (transitionsTotal_tri[prevtag1, prevtag2] -1)
		else:
			num_tri = 1.0 * (transitions_tri[prevtag1, prevtag2][tag]) / (transitionsTotal_tri[prevtag1, prevtag2])
		num_bi = 1.0 * (transition_bi[prevtag2][tag] - 1) / (transitionsTotal_bi[prevtag2] - 1) 
		num_uni = 1.0 * (transition_uni[tag] - 1) / (tagTotal - 1)
		max_num = max(num_uni, num_bi, num_tri)
		if max_num == num_tri:
			lambda_3 += transitions_tri[prevtag1, prevtag2][tag]
		if max_num == num_bi:
			lambda_2 += transitions_tri[prevtag1, prevtag2][tag]
		if max_num == num_uni:
			lambda_1 += transitions_tri[prevtag1, prevtag2][tag]

total = lambda_1 + lambda_2 + lambda_3
lambda_1 = 1.0 * lambda_1 / total
lambda_2 = 1.0 * lambda_2 / total
lambda_3 = 1.0 * lambda_3 / total

for pretag1 in transition_uni:
	for pretag2 in transition_uni:
		for tag in transition_uni:
			if (pretag1, pretag2) not in transitions_tri:
				transitions_tri[(pretag1, pretag2)] = defaultdict(int)
			if tag not in transitions_tri[(pretag1,pretag2)]:
				transitions_tri[(pretag1, pretag2)][tag] = 1e-12

for (prevtag1, prevtag2) in transitions_tri:
	for tag in transitions_tri[prevtag1, prevtag2]:
		if transitionsTotal_tri[prevtag1, prevtag2] != 0:
			prob_estimated = lambda_3 * (transitions_tri[prevtag1, prevtag2][tag]) / transitionsTotal_tri[prevtag1, prevtag2] 
		else:
			prob_estimated = 0.0
		if prevtag2 in transition_bi:
			prob_estimated += lambda_2 * transition_bi[prevtag2][tag] / transition_uni[prevtag2]
		prob_estimated += lambda_1 * transition_uni[tag] / tagTotal 
		prob_estimated += lambda_1 * transition_uni[tag] / tagTotal 

		print "trans %s %s %s %s" % (prevtag1, prevtag2, tag, float(prob_estimated))

for tag in emissions:
	for token in emissions[tag]:
		print "emit %s %s %s" % (tag, token, float(emissions[tag][token]) / emissionsTotal[tag])



