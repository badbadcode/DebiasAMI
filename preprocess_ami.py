# the text in social media platform is irregular
import pickle
import pandas as pd
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import preprocessor as p
from tqdm import tqdm
from utils.config import Config
from utils.data_class import Counterfactual


def remove_urls(vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\-|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    vTEXT = re.sub(r'(www)(\w|\.|\-|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return (vTEXT.lower())


# %%

no_abbre = {
    "ain't": "have not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "I would",
    "i'd": "I had",
    "i'll": "I will",
    "i'm": "I am",
    "isn't": "is not",
    "it's": "it is",
    "it'll": "it will",
    "i've": "I have",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "weren't": "were not",
    "we've": "we have",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "youre": "you are",
    "you've": "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll": " will",
    "didn't": "did not",
    "tryin'": "trying",
    "pls": "please"
}

# %%

emoji = {
    "&lt;3": " good ",
    ":d": " good ",
    ":dd": " good ",
    ":p": " good ",
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":s": " bad ",
    ":-s": " bad ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha"}

print("....start....cleaning")

# =================stop word=====================

lem = WordNetLemmatizer()

stop_words = ['the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this', 'that', 'these',
              'those', 'then',
              'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during', 'to']


eng_stopwords = set(stopwords.words("english"))

# =================replace the duplicate word=====================


def substitute_repeats_fixed_len(text, nchars, ntimes=3):
    """
         Find substrings that consist of `nchars` non-space characters
         and that are repeated at least `ntimes` consecutive times,
         and replace them with a single occurrence.
         Examples:
         abbcccddddeeeee -> abcde (nchars = 1, ntimes = 2)
         abbcccddddeeeee -> abbcde (nchars = 1, ntimes = 3)
         abababcccababab -> abcccab (nchars = 2, ntimes = 2)
    """
    return re.sub(r"(\S{{{}}})(\1{{{},}})".format(nchars, ntimes - 1), r"\1", text)


def substitute_repeats(text, ntimes=3):
    # Truncate consecutive repeats of short strings
    for nchars in range(1, 20):
        text = substitute_repeats_fixed_len(text, nchars, ntimes)
    return text


# =================choose one of tokenizer=======================

# tokenizer = RegexpTokenizer(r'\w+')  # tokenizer defined by your re rules
tokenizer = TweetTokenizer() # a tokenizer to label text in social media



# =================final clean function=======================
def clean(comment, remove_stopwords=True, remove_punctuations=False):
    # comment=re.sub("\.\."," .",comment)
    comment = remove_urls(comment.lower())
    # remove \n
    comment = re.sub(r"\t", " ", comment)
    comment = re.sub(r"\r\n", " . ", comment)
    comment = re.sub(r"\r", " . ", comment)
    comment = re.sub(r"\n", " . ", comment)
    comment = re.sub(r"\\n\n", " . ", comment)
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", comment)
    comment = re.sub("\[\[.*\]", "", comment)
    comment = re.sub(r"\'ve", " have ", comment)

    comment = re.sub(r"\'d", " would ", comment)
    comment = re.sub(r"\'ll", " will ", comment)
    comment = re.sub(r"ca not", "cannot", comment)
    comment = re.sub(r"you ' re", "you are", comment)
    comment = re.sub(r"wtf", "what the fuck", comment)
    comment = re.sub(r"i ' m", "I am", comment)
    comment = re.sub(r"I", "one", comment)
    comment = re.sub(r"II", "two", comment)
    comment = re.sub(r"III", "three", comment)
    comment = re.sub(r"mothjer", "mother", comment)
    comment = re.sub(r"nazi", "nazy", comment)
    comment = re.sub(r"withought", "with out", comment)
    comment = substitute_repeats(comment)
    s = comment

    s = s.replace('&', '')
    s = s.replace('@', '')
    s = s.replace('0', '')
    s = s.replace('1', '')
    s = s.replace('2', '')
    s = s.replace('3', '')
    s = s.replace('4', '')
    s = s.replace('5', '')
    s = s.replace('6', '')
    s = s.replace('7', '')
    s = s.replace('8', '')
    s = s.replace('9', '')
    # s = s.replace('雲水','')

    comment = s
    # comment = re_tok.sub(' ', comment)
    # print(comment)
    words = tokenizer.tokenize(comment)
    words = [no_abbre[word] if word in no_abbre else word for word in words]
    words = [emoji[word] if word in emoji else word for word in words]
    if remove_stopwords:
        words = [w for w in words if not w in stop_words]

    sent = " ".join(words)
    # Remove some special characters, or noise charater, but do not remove all!!
    if remove_punctuations:
        sent = re.sub(r'([\'\"\/\-\_\--\_])', ' ', sent)
    else:
        sent = re.sub(r'([\'\"\/\-\_\-\_\(\)\{\}])', ' ', sent)
    clean_sent = re.sub(r'([\;\|•«\n])', ' ', sent)
    clean_sent = re.sub(r"n't", " not ", clean_sent)

    FLAG_remove_non_ascii = True
    if FLAG_remove_non_ascii:
        return clean_sent.encode("ascii", errors="ignore").decode().strip()
    else:
        return clean_sent.strip()

def clean_split(split):
    path = Config.OLD_DATA_DIC["AMI"][split]

    if split == "unbiased":
        pd_train_binary = pd.read_csv(path, sep='\t',names=["text","misogynous"])
    else:
        eng_train_dataset = pd.read_csv(path, sep='\t')
        pd_train_binary = eng_train_dataset[['id', 'misogynous', 'text', 'misogyny_category', 'target']]
    cleaned_comments = []
    for i in tqdm(range(len(pd_train_binary))):
        comment = pd_train_binary.iloc[[i][:]]
        tweet_cleaned = p.clean(comment['text'][i])
        cleaned = clean(tweet_cleaned, remove_stopwords=False, remove_punctuations=False)
        cleaned_comments.append(cleaned)
        print("UNCLEANED_TEXT:", comment['text'][i])
        print("CLEANED_TEXT:", cleaned)
        # print("CLEANED_TEXT:", clean(comment['text'][i], remove_stopwords=False, remove_punctuations=False))
        # print("CLEANED_TEXT_PREPROCESSOR:", p.clean(comment['text'][i]))
        l = p.parse(comment['text'][i])
        print("ALL_mention:", l.mentions)
        print("ALL_url:", l.urls)
        print("ALL_hashtags:", l.hashtags)
        print("ALL_emojis:", l.emojis)
        print("ALL_smiley:", l.smileys)
        print("ALL_smiley:", l.smileys)
        print("ALL_number:", l.numbers)
        print("\n\n")
    pd_train_binary.rename(columns={"misogynous": "label", "text":"old_text"})
    pd_train_binary["text"] = cleaned_comments
    return pd_train_binary

ami_class = Counterfactual(clean_split("train"),
                      clean_split("test"),
                      clean_split("unbiased"),
                      moniker="AMI")
pickle.dump(ami_class, open(Config.DATA_DIC["AMI"], "wb"))


# lines = open("./resource/gender list/generalized_swaps.txt").readlines()
# baseline_swap_list = [["woman", "man"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"]]
# female_words = [x[0] for x in baseline_swap_list]




