import nltk
from urllib.request import urlopen
link = "https://byjus.com/kids-learning/stories/"
html_content = urlopen(link).read()
print(html_content)

from bs4 import BeautifulSoup
soup = BeautifulSoup(html_content)
print(soup)

for s in soup(['script','style']):
    s.extract()

text = soup.get_text()
text = " ".join(text.split())
print("\n",text)

text = text.lower()
print("Text after lower:\n", text)

# Word tokenize

from nltk.tokenize import sent_tokenize, word_tokenize
print("Sentence tokenize :\n", sent_tokenize(text))
text_nltk = word_tokenize(text)
print("Text after word tokenize:\n", text_nltk)

our_stopwords = {'is','i','the','.','am','of','are','it','they','as'}
text2 = text_nltk
for w in list(text2):
    if w in our_stopwords:
        count_w = text2.count(w)
        for j in range(count_w):
            text2.remove(w)
print("Text after removal of stopwords:\n", text2)

# Download nltk stopwords and remove them from text

from nltk.corpus import stopwords
nltk_stopwords = set(stopwords.words("english"))
nltk_stopwords.remove("but")
nltk_stopwords.add("-")
text2 = text_nltk
for w in list(text2):
    if w in nltk_stopwords:
        count_w = text2.count(w)
        for j in range(count_w):
            text2.remove(w)
print("Text after removal of nltk stopwords:\n", text2)

# Find tag of each words
text4 = nltk.pos_tag(text2)
print("POS tag of each word before stemming:\n", text4)

# Stemming

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
text3 = [stemmer.stem(word) for word in text2]
print("Text after stemming:\n", text3)

# Tag of each word after stemming

text5 = nltk.pos_tag(text3)
print("POS tag of each word after stemming:\n", text5)

# Default POS tag
from nltk.tag import DefaultTagger
py_tag = DefaultTagger("NN")
tag_txt = py_tag.tag(text3)
print("After default tag:\n", tag_txt)

# Text 4 is list of tuple
JJ_count = 0
adjs = set({'dummy'})
for word_set in list(text4):
    if word_set[1] == 'JJ':
        JJ_count += 1
        adjs.add(word_set[0])

print("Total set of adjectives:\n", JJ_count)
print("Unique set of adjectives used in text :\n", adjs)

# Building wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wc = WordCloud(max_words = 200, background_color = 'black').generate(" ".join(text2))
plt.imshow(wc)
plt.show()

