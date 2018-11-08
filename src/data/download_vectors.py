"""Download vectors from links in table."""

import pandas as pd
import urllib as ur


def main(path_to_table):
	"""Load links from .csv file, and save list of languages

	Parameters
	----------
	path_to_table: str
	  path to .csv file with fasttext links
	"""

	# Open up .csv file
	df_languages = pd.read_csv(path_to_table)

	for index, row in df_languages.iterrows():
		language = row['language']
		text_link = row['link']
		# if language == "zulu":
		response = ur.urlopen(text_link)
		file_obj = response.read()
		with open("data/vectors/{language}.vec".format(language=language), "wb") as f:
			f.write(file_obj)


if __name__ == "__main__":
	from argparse import ArgumentParser 

	parser = ArgumentParser()

	parser.add_argument("--table-path", type=str, dest="path_to_table",
						default="data/raw/all_languages.csv")
	args = vars(parser.parse_args())
	print(args)
	main(**args)