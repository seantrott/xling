"""Read in .vec file without gensim."""


from scipy import spatial


def read_vec_file(path):
	"""Read in .vec file."""
	f = open("data/vectors/zulu.vec", "rb")

	vocab = dict()
	for line in open(path, "r"):
	    fields = line.strip().split(" ") 
	    word = fields[0]
	    vec = [float(v) for v in fields[1:]]
	    vocab[word] = vec

	return vocab

def meaning_similarity(vocab, w1, w2):
	"""Utility function."""
	return compute_similarity(vocab[w1], vocab[w2])

def compute_cosine_distance(w1, w2):
	"""Return cosine distance."""
	return spatial.distance.cosine(w1, w2)

def compute_similarity(w1, w2):
	return 1 - compute_cosine_distance(w1, w2)

