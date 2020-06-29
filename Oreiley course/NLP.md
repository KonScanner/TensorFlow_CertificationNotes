### Explore Word Analogies

- Word2Viz:
  - https://lamyiowce.github.io/word2viz/

### Word Representations

#### One-hot vs Vector-based

- _One-hot_

  - Not subtle
  - Manual curated taxanomies
  - Handles new words poorly
  - Subjective
  - Word similarity is not represented

- _Vector-based_
  - Very nuanced
  - Automatic
  - Seamlessly incorporates new words
  - Driven by natural language data
  - Word similarity == proximity in space

## Elements of Natural Human Language

- _Morphenes_ and _Phonemes_ that get **combined** to produce _words_ which are then **combined** to produce _syntax_ which is the **combined** to create _semantics_. With **increasing** complexity moving from _Morphenes_ to _semantics_.

  - https://en.wikipedia.org/wiki/Morpheme
  - https://en.wikipedia.org/wiki/Phoneme

|     Representation      |         Traditional ML         | Deep Learning | Audio Only |
| :---------------------: | :----------------------------: | :-----------: | :--------: |
|  Phonology (e.g. hat)   | All phonemes ('h','a' and 't') |    Vectors    |    True    |
| Morphology (e.g. manly) | All morphemes ('man' as base)  |    Vectors    |   False    |
|          Words          |        One-hot encoding        |    Vectors    |   False    |
|         Syntax          |          Phrase rules          |    Vectors    |   False    |
|        Semantics        |        Lambda calculus         |    Vectors    |   False    |

#### Word2Vec:

|  Architecture  |                 Predicts                  |                       Relative Strengths                        |
| :------------: | :---------------------------------------: | :-------------------------------------------------------------: |
| Skip-gram (SG) |      Context words given target word      |      Better for smaller corpus represents rare words well       |
|      CBOW      | Target word given context word(s) average | Multiple times faster represents frequent words slightly better |

#### GloVe:

- Global vectors to word representations, is count based. Designed to be parallelizable over multiple corpus samples and machines.

#### FastText:

- Works on sub-word level, its 'word-vectors' are actually sub-components of words. This enables it to work around the related issues with rare words or words that show up at inference time, that were not present in the training set.

#### Language Embedding Levels

- Word
- Subword (fastText)
- Character
- Document (doc2vec)
  - Useful when you're trying to gauge document similarity.

#### Pre-trained Word Vectors:

- word2vec: github.com/Kyubyong/wordvectors
- GloVe: nlp.stanford.edu/projects/glove
- fastText: fasttext.cc
- https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

### Evaluating word vectors:

- ### Extrinsic
  - Involve assesing the performance of your word vectors within whaterver your downstream application of interest is. (e.g. your sentiment classifier or otherwise). Takes longer to carry out, requires to carry out all downstream processing steps before evaluation.
- ### Intrinsic
  - Involves assesing the performance of your word vectors not on your final nlp application, but on some intermediate subtask. One common sub-task is assessing on whether your word vectors correspond well to arethmetical analogies.

## Pre-processing Natural Language Data

- _Tokenization_:

  - Splitting of a document (e.g. book), into a list of discrete elements of language (e.g. words), which we call elements.

- _To lowercase_: She -> she

  - Normalizing "meaning" in the vector space
  - An example with "General Potato" and "general potato" may break the model, so depending on your application, make the right decision on whether to use it or not.

- _Removing "stopwords"_:

  - "That", "Which", "of", contain little to no meaning, they can be removed. Depending on your application you may want to whitelist or blacklist certain words.

- _Removing punctuation_:

  - They don't add extra value to an NLP model, so they are usually removed.

- _Stemming_:

  - The truncation of words, down to their "stem". Particularly useful with smaller datasets, as it pulls words with the same meaning in the same pool efficiently.
  - (house -> hous; housing -> hous)

- _Lemmetization_:

  - Requires a reference dictionary and has a similar effect to stemming. It will provide more nuanced results though.
  - (are -> be; housing -> house)

- _Handling n-grams_:
  - Some words commonly occur in such a way that their combination of words is better to be considered as a single concept, rather than its individual components contexts.
  - `[new, york]` -> `[new_york]`
