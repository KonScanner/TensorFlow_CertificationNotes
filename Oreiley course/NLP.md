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

### Elements of Natural Human Language

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

### Evaluating word vectors:

- ### Extrinsic
  - Involve assesing the performance of your word vectors within whaterver your downstream application of interest is. (e.g. your sentiment classifier or otherwise). Takes longer to carry out, requires to carry out all downstream processing steps before evaluation.
- ### Intrinsic
  - Involves assesing the performance of your word vectors not on your final nlp application, but on some intermediate subtask. One common sub-task is assessing on whether your word vectors correspond well to arethmetical analogies.
