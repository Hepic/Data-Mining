#!/usr/bin/env python
"""
Minimal Example
===============
Generating a square wordcloud from the US constitution using default arguments.
"""
import csv
from os import path
from wordcloud import WordCloud

#d = path.dirname(__file__)
#Read the whole text.
#text = open(path.join(d, 'my.txt')).read()
category=["Politics","Film","Football","Business","Technology"]
with open('train_set.csv','rb') as f:
	reader=csv.reader(f)
	text = " ".join([" ".join(value[1:]) for value in category.values()])






	# Generate a word cloud image
	wordcloud = WordCloud().generate(text)

	# Display the generated image:
	# the matplotlib way:
	import matplotlib.pyplot as plt
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")

	# lower max_font_size
	wordcloud = WordCloud(max_font_size=100).generate(text)
	plt.figure()
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.show()

# The pil way (if you don't have matplotlib)
# image = wordcloud.to_image()
# image.show()
