from datasets import load_dataset, Dataset
import pandas as pd
import os
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import wordnet
import random
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download("punkt")


def getSynonym(word):
    synonyms = []
    for synonym in wordnet.synsets(word):
        for lemma in synonym.lemmas():
            if lemma.name() != word:
                synonyms.append(lemma.name())
    return synonyms


def synonymReplace(text, index):
    words = word_tokenize(text)
    taggedWords = pos_tag(words)
    newWords = words.copy()

    words.copy()

    for i, (word, tag) in enumerate(taggedWords):
        if tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ'):  # Replaces nouns, adjectives, and verbs
            synonyms = getSynonym(word)
            if synonyms:
                if len(synonyms) > i:
                    newWords[i] = synonyms[index]
                else:
                    newWords[i] = synonyms[0]
    return ' '.join(newWords)


def randomInsert(text, keywords, n=1):
    words = word_tokenize(text)
    for i in range(n):
        addWord = random.choice(keywords)
        insertPosition = random.randint(0, len(words))
        words.insert(insertPosition, addWord)
    return ' '.join(words)


def randomSwap(text, n=1):
    words = word_tokenize(text)
    for i in range(n):
        index1, index2 = random.sample(range(len(words)), 2)
        words[index1], words[index2] = words[index2], words[index1]
    return ' '.join(words)


def randomDelete(text, p=0.1):
    words = word_tokenize(text)
    newWords = [word for word in words if random.random() > p]
    return ' '.join(newWords)


def augmentData(df):
    # Randomly insert keywords to increase weight
    # Tweak n to increase or decrease how many keywords are added
    keywords = df['keywords'].tolist()
    df['randomInserts'] = df['ancient_text'].apply(lambda text: randomInsert(text, keywords, n=3))

    # Synonym replacement, word swapping, and random deletes
    for i in range(0,3):
        df[f'synonyms{i}'] = df['randomInserts'].apply(lambda text: synonymReplace(text, i))
        df[f'swaps{i}'] = df[f'synonyms{i}'].apply(lambda text: randomSwap(text, n=15))
        df[f'deletes{i}'] = df[f'synonyms{i}'].apply(lambda text: randomDelete(text, p=.1))

    df.to_csv('augmented.csv', index=False)


def main():
    if not os.path.exists("original.csv"):
        # https: // huggingface.co / datasets / Svngoku / ancient_egypt_pyramid_text_dataset?row = 1
        dataset = load_dataset("Svngoku/ancient_egypt_pyramid_text_dataset")
        df = pd.DataFrame(dataset['train'])
        df.to_csv('original.csv', index= False)
        print(f"Original Columns: {df.columns}")
        print(f"The original dataset has {dataset['train'].num_rows} rows of data and {dataset['train'].num_columns} columns of data")
        print(f"The original dataset is {os.path.getsize('original.csv')}")
    else:
        df = pd.read_csv("original.csv")
        print(f"Augmented Columns: {df.columns}")
        print(f"The original dataset has {df.shape[0]} rows of data and {df.shape[1]} columns of data")
        print(f"The original dataset is {os.path.getsize('original.csv')} bytes")

    if os.path.exists("augmented.csv"):
        os.remove("augmented.csv")

    augmentData(df)

    aug = pd.read_csv("augmented.csv")
    print(f"\n\nAugmented Columns: {aug.columns}")
    print(f"The augmented dataset has {aug.shape[0]} rows of data and {aug.shape[1]} columns of data")
    print(f"The augmented dataset is {os.path.getsize('augmented.csv')} bytes")


main()