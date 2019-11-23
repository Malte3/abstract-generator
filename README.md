[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Malte3/abstract-generator/blob/master/abstract-generator.ipynb)

# Paper Abstract Generator by a Sequence to Sequence Network [WIP]

This repository contains code for in-progress implementations and experiments for generating paper abstracts from data downloaded from the open access electronic preprint database [arXiv](https://arxiv.org/).

## Creating the Dataset

First the metadata, including the abstract, is downloaded with a http requests through the [arXiv API](https://arxiv.org/help/api). The papers are downloaded as Pdf using the included link in the metatdata and stored temporarily to extract the plain text. The plain text is extracted using a Python port of the Apache Tika library [tika-python](https://github.com/chrismattmann/tika-python). Most of the papers on arxiv have the abstract as first paragraph in the paper, because the task here is not to extract the first paragraph this part needs to be removed. Unfortunately the abstract in the metadata and the one in the paper sometimes have big differences. Because of this first all line breaks are removed and words that are hyphenated because of a line break are merged. To remove the abstract then three attempts are taken:

1. Remove the Paragraph with the heading "Abstract" and ending with a double line break that has approximately the same length as the abstract from the metadata.

2. Remove the Paragraph that starts and ends with the same three letters and has approximately the same length as the abstract from the metadata.

3. Remove the Paragraph that starts and ends with the same word and has approximately the same length as the abstract from the metadata.

These operations are done using regular expressions, in case that the heading "Abstract" was not removed before it is also removed. This can not guarantee that the abstract was remove from the paper or a wrong part was removed

To put a limit to the computation time and memory consumption of the later computations the plain text of the paper is reduced by a python implementation of TextRank ([summa – textrank](https://github.com/summanlp/textrank)). With that a approximate number of output words can be generated that consist the most important sentences and words in the text.

Further preprocessing is done by the dataset classes of [torchtext](https://pytorch.org/text/index.html). These split the text into words, by tokenization and build a vocabulary. Furthermore, the way of numericalization of the text is initialized by the word embedding from the Global Vectors for Word Representation [GloVe](https://nlp.stanford.edu/projects/glove/).

## The Sequence to Sequence Model

After the data was prepared, we can now train the sequence to sequence model. The model consists of an encoder and a decoder network. The encoder encodes the condensed information, so that the decoder can use this information to put out human readable text that summarizes the given input text. In contrast to other summarizing approaches this should not just extract important sentences and words, but give the abstract information of the text. The encoder consists of a simple recurrent neural network and the decoder consists of a recurrent neural network with attention. This model is from the tutorial [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html), also the training procedure from that tutorial were adapted so the new dataset can be used.

## Example

These Examples were produced by training on only 3 samples with maximum length of 500 words. More experiments have to be done.

### Preprocessed Original Abstract
, a higher , has been in [22] , which is shown that have distance with more number of than a single ej . some such as and are well known and used in . in , a a to every other in the . , in , every is as a
and its to every other in the . in this paper , an in higher ej is . the paper shows that the a lower number of to the . in , since the are assumed to be , the is into three . the are and that the than the and has 2 . total number of

### Preprocessed Paper
4 . 1 the previous and the have the same number of and , which is where m is the of the and n is the number of . 1 1 ( , ) 2 let n be the number of and m be the of the 3 via ( , , , , , ) s1 4 via ( , , ,
m − 1 , m − 1 , 1 ) s2 5 via −1 ( −1 , , , m − 1 , m − 1 , 1 ) 6 via ( , −1 , , m − 1 , m − 1 , 1 ) 7 via ( , , , m − 1 , m − 1 , 1 ) 8 via ( , , , m − 1 , m − 1 , 1 ) 9 if > 1 then 10 ( 1 , ) 4 . 2 the the in every in the . 9 2 1 ( , , , x , y , ) 2 let n be the number of and m be the of the 3 if > then 4 5 if x > 0 then 6 via ( , , , x− 1 , 0 , 1 ) 7 if y > 0 then 8 via ( , , , x− 1 , y − 1 , 1 ) 9 if > 1 then 10 ( 1 , ) 3 1 ( , , phase ) 2 let n be the number of and m be the of the 3 if phase = 1 then 4 via ( , , , m − 1 , m − 1 , 1 , phase ) 5 via ( , , , m − 1 , m − 1 , 1 , phase ) 6 if phase = 2 then 7 via ( , , , m − 1 , m − 1 , 1 , phase ) 8 via −1 ( −1 , , , m − 1 , m − 1 , 1 , phase ) 9 if phase = 3 then 10 via ( , −1 , , m − 1 , m − 1 , 1 , phase ) 11 via ( , , , m − 1 , m − 1 , 1 , phase ) 12 if > 1 then 13 ( 1 , , phase ) 10 figure 7 1 of the in ej ( 2 ) . in , all 11 4 1 ( , , , x , y , , phase ) 2 let n be the number of and m be the of the 3 if > and phase = 3 then 4 5 if > and phase < 3 then 6 ( n , 1 , ) 7 8 if x > 0 then 9 via ( , , , x− 1 , 0 , 1 , phase ) 10 if y > 0 then 11 via ( , , , x− 1 , y − 1 , 1 , phase ) 12 if > 1 then 13 ( 1 , , 14 phase ) open the other three and to the from their over the ( , −1 , and ) . note that , the previous the in where n is the number of and m is the ( and the number of in each ) . then , the number of in each for the previous is 14 1 an of the ( previous ) on ej ( 3 ) 1 1 50 , 1 6 7
2 50 , 6 12 18 3 50 , 12 18 30 2 1 50 , 2 , 3 , 1 , 3 1 , 1 , 8 , 9 , 2 , 8 , 16 , , 3 9 , 16 , , , total 12 , 50 , and as = ( α ) ( 5 ) where r is the number such that 0 ≤ r < n and d is the distance or number r such that 1 ≤ d ≤ m . 1 the number of , , , and in each in the ej ( 3 ) for the previous . 2 the number of , , , and in each of the applied on the ej ( 3 ) . figure describes the number of in each of the in ej ( 3 ) . from and 18 , it can be that the in the is between the and in the 18 0 1 2 3 4 5 6 7 8 9 n u m b o f n o d previous figure number of in each in ej ( 3 ) . 0 1 2 3 4 5 6 7 8 9 n u m b o f n o d previous figure 18 number of in each in ej ( 3 ) . the total number of for all of the in ej ( n ) for n = 1 to 6 is in 3 for both , the previous and the , .

### Network Outpt 

all . particular , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
## Problems

- A lot of technical terms, that become out of vocabular words
    - Therefore, use more general paper or texts with simpler language
    - Manually prepare the dataset (very time consuming)
- Also a lot of formulas and numbers that also become out of vocabular words
- Training process needs to be optimized
