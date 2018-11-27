#!/usr/bin/python3

#SBATCH --output=output.log
#SBATCH --partition=Comet
#SBATCH -n 16 # 5 cores

import os
import random

import pandas as pd

import sys

import editdistance as ed
import scipy.stats as ss

from joblib import Parallel, delayed

from itertools import combinations
from tqdm import tqdm

from gensim.models import KeyedVectors


num_cores = int(sys.argv[1])
NUM_JOBS = num_cores


def contains_html(word):
	"""Return whether word contains .html characters."""
	html_characters = ['<', '[', '/', '(', '>']
	return any(elem in word for elem in html_characters)

def remove_html_words(words):
	"""Remove all words in list with .html characters."""
	filtered = [w for w in words if not contains_html(w)]
	return filtered


def compare_form_and_meaning(w1, w2):
	return {'orthographic_distance': ed.eval(w1, w2),
			'meaning_similarity': model.similarity(w1, w2),
			'w1': w1,
			'w2': w2}

def get_comparisons(model, words):
	analysis = Parallel(n_jobs=NUM_JOBS, verbose=10)(delayed(compare_form_and_meaning)(w1, w2) for w1, w2 in combinations(words, 2))

	return pd.DataFrame(analysis)


df_languages = pd.read_csv("data/raw/all_languages.csv")

for language in df_languages['language']:
	outfile_path = "data/processed/parallelized_analysis/{language}_results.csv".format(language=language)
	if os.path.exists(outfile_path):
		print("Already analyzed for {language}".format(language=language))
		continue
	try:
		print("Trying to load model for {language}...".format(language=language))
		filepath = 'data/vectors/{language}.vec'.format(language=language)
		model = KeyedVectors.load_word2vec_format(filepath)
		# vocab_mappings = read_vec_file(filepath)
		print("Loaded model for {language}...".format(language=language))
	except Exception as e:
		print(e)
		print("File for '{language}' not found! Moving on to the next language.".format(language=language))
		continue
	
	print("Now conducting analysis for systematicity in {language}...".format(language=language))

	# Preprocess vocab to remove .html characters, etc.
	words = model.vocab.keys()
	words = remove_html_words(words)

	num_words = len(words)
	print("Vocab size: {n}".format(n=num_words))

	df_comparisons = get_comparisons(model=model, words=words)

	reg = ss.linregress(df_comparisons['orthographic_distance'],
						df_comparisons['meaning_similarity'])

	output = [{'vocab_size': num_words,
			  'slope': reg.slope,
			  'intercept': reg.intercept,
			  'correlation': reg.rvalue,
			  'pval': reg.pvalue,
			  'std': reg.stderr}]

	df_output = pd.DataFrame(output)

	df_output.to_csv("data/processed/parallelized_analysis/{language}_results.csv".format(language=language))
	# df_comparisons.to_csv("data/processed/{language}_comparisons.csv".format(language=language))
