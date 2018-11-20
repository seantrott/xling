"""Read in vectors for each language.

TO DO:
- Parallelize
- Ignore/remove .html items in vocab
- Should we take negative sign of form or meaning distance (to make them compatible / same direction?)

TO SAVE:
- vocab size
- results of regression analysis
- all the actual comparisons? (To use in a full mixed-model analysis)

"""

import os
import random

import pandas as pd 
import editdistance as ed
import scipy.stats as ss

from itertools import combinations
from tqdm import tqdm

from gensim.models import KeyedVectors

from vector_utils import read_vec_file, compute_cosine_distance, compute_similarity


def contains_html(word):
	"""Return whether word contains .html characters."""
	html_characters = ['<', '[', '/', '(', '>']
	return any(elem in word for elem in html_characters)

def remove_html_words(words):
	"""Remove all words in list with .html characters."""
	filtered = [w for w in words if not contains_html(w)]
	return filtered

def compare_form_and_meaning(model, words):
	"""Compare form and meaning distances of all word pairs."""
	# combos = list(combinations(words, 2))

	analysis = []
	# for w1, w2 in tqdm(combos):
	for w1, w2 in combinations(words, 2):
		orth_distance = ed.eval(w1, w2)
		meaning_similarity = model.similarity(w1, w2)
		analysis.append({
			'w1': w1,
			'w2': w2,
			'orthographic_distance': orth_distance,
			'meaning_similarity': meaning_similarity
			})

	return pd.DataFrame(analysis)



df_languages = pd.read_csv("data/raw/all_languages.csv")

for language in df_languages['language'][2:-1]:
	outfile_path = "data/processed/{language}_results.csv".format(language=language)
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
	# words = vocab_mappings.keys()
	words = remove_html_words(words)

	num_words = len(words)
	print("Vocab size: {n}".format(n=num_words))

	# df_comparisons = compare_form_and_meaning(vocab_mappings=vocab_mappings, words=words)

	df_comparisons = compare_form_and_meaning(model=model, words=words)

	reg = ss.linregress(df_comparisons['orthographic_distance'],
						df_comparisons['meaning_similarity'])

	output = [{'vocab_size': num_words,
			  'slope': reg.slope,
			  'intercept': reg.intercept,
			  'correlation': reg.rvalue,
			  'pval': reg.pvalue,
			  'std': reg.stderr}]

	df_output = pd.DataFrame(output)

	df_output.to_csv("data/processed/{language}_results.csv".format(language=language))
	# df_comparisons.to_csv("data/processed/{language}_comparisons.csv".format(language=language))
