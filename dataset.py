import urllib.request
import lxml.etree as etree
import tika
from tika import parser
import csv
import os
import re
import shutil
import torch

import torchtext
from torchtext.data.utils import get_tokenizer, RandomShuffler
from torchtext.data.dataset import check_split_ratio, rationed_split, stratify
from summa import summarizer

tika.initVM()

class PaperAbstractDataset(torchtext.data.Dataset):
    """Defines a dataset composed of Examples along with its Fields, for paper and abstracts.
    """
    sort_key = None

    @classmethod
    def splits(cls, search_query = 'all', max_results = 300, start = 0, reduced_words=1000, savepath='.data',  split_ratio=0.7, stratified=False, strata_field='abstract',
              random_state=None, **kwargs):
        """Create Dataset objects for multiple splits of a dataset.

        Arguments:
            search_query (str): specify searchh query for arxiv, default is 'all' results, more information on ttps://arxiv.org/help/api.
            max_results (int): maxium search results from arxiv from search query
            savepath (str): save path for the txt files
            split_ratio (float or List of floats): a number [0, 1] denoting the amount
                of data to be used for the training split (rest is used for validation),
                or a list of numbers denoting the relative sizes of train, test and valid
                splits respectively. If the relative size for valid is missing, only the
                train-test split is returned. Default is 0.7 (for the train set).
            stratified (bool): whether the sampling should be stratified.
                Default is False.
            strata_field (str): name of the examples Field stratified over.
                Default is 'label' for the conventional label field.
            random_state (tuple): the random seed used for shuffling.
                A return value of `random.getstate()`.

        Returns:
            Tuple[Dataset]: Datasets for train, validation, and
            test splits in that order, if provided.
        """

         # initialize text field
        text_field = torchtext.data.Field(tokenize=get_tokenizer("spacy"), init_token='<sos>', eos_token='<eos>', lower=True)
        fields = [('abstract', text_field), ('paper', text_field)]
        examples = []

        # Create new dataset by downloading from arxiv or open dataset from folder
        if not os.path.exists(savepath):
            os.mkdir(savepath)
            # create directories for saving the data set
            if not os.path.exists(os.path.join(savepath, 'temp')):
                os.mkdir(os.path.join(savepath, 'temp'))
            if not os.path.exists(os.path.join(savepath, 'abstracts')):
                os.mkdir(os.path.join(savepath, 'abstracts'))
            if not os.path.exists(os.path.join(savepath, 'paper')):
                os.mkdir(os.path.join(savepath, 'paper'))
            data = cls.download(search_query=search_query, max_results=max_results)
            abstracts, papers = cls.extract_paper_and_abstract(data, savepath=savepath)

            # generate all examples
            for i, (abstract, paper) in enumerate(zip(abstracts, papers)):
                paper_tokenized = []
                abstract_tokenized = []
                # rdeduce the number of words with the textrank approach
                textranked_paper = summarizer.summarize(paper, words=reduced_words)
                # add start and end token
                paper_tokenized += [u'<sos>'] + text_field.preprocess(textranked_paper) + [u'<eos>']
                abstract_tokenized += [u'<sos>'] + text_field.preprocess(abstract) + [u'<eos>']
                # initialize examples
                examples.append(torchtext.data.Example.fromlist([abstract_tokenized, paper_tokenized], fields))

                # save data samples in txt files
                with open(os.path.join(savepath,'abstracts','abstract_' + str(i) + '.txt'), 'w+', encoding='utf-8') as abstr:
                    csvwriter = csv.writer(abstr, delimiter=' ')
                    csvwriter.writerow(abstract_tokenized)
                with open(os.path.join(savepath,'paper','paper_' + str(i) + '.txt'), 'w+', encoding='utf-8') as pap:
                    csvwriter = csv.writer(pap, delimiter=' ')
                    csvwriter.writerow(paper_tokenized)

        else:
            # read all files in saved data path
            paper_files = os.listdir(os.path.join(savepath,'paper'))
            abstract_files = os.listdir(os.path.join(savepath,'abstracts'))
            data = [[paper, abstract] for paper, abstract in zip(paper_files, abstract_files)]
            papers_tokenized = []
            abstracts_tokenized = []
            # read paper and abstracts from files
            for paper, abstract in data:
                with open(os.path.join(savepath, 'paper', paper), encoding='utf-8') as csvfile:
                    paper = csv.reader(csvfile, delimiter=' ')
                    for pap in paper:
                        if pap:
                            papers_tokenized.append(pap)
                with open(os.path.join(savepath, 'abstracts', abstract), encoding='utf-8') as csvfile:
                    abstract = csv.reader(csvfile, delimiter=' ')
                    for abstr in abstract:
                        if abstr:
                            abstracts_tokenized.append(abstr)
       
            # generate all examples
            for abstract_tokenized, paper_tokenized in zip(abstracts_tokenized, papers_tokenized):
                examples.append(torchtext.data.Example.fromlist([abstract_tokenized, paper_tokenized], fields))

        # create initial dataset
        dataset = PaperAbstractDataset(examples, fields)
        # split dataset
        splits = dataset.split(split_ratio=split_ratio, stratified=stratified, strata_field=strata_field,
              random_state=random_state)
        # initialize vocabulary
        pre_trained_vector_type = 'glove.6B.300d'
        for d in splits:
            for name, field in d.fields.items():
                field.build_vocab(splits[0], vectors=pre_trained_vector_type)
            d.filter_examples(['abstract', 'paper'])
        return splits

    @classmethod
    def download(cls, search_query = 'all', max_results = 300, start = 0):
        '''
            Download e-prints from https://arxiv.org with arXiv API
            search_query (str): specify searchh query for arxiv, default is 'all' results, more information on ttps://arxiv.org/help/api.
            max_results (int): maxium search results from arxiv from search query
        '''
        url = 'http://export.arxiv.org/api/query?search_query=' + search_query + '&start=' + str(start) + '&max_results=' + str(max_results)
        data = urllib.request.urlopen(url).read()
        return data

    @classmethod
    def extract_paper_and_abstract(cls, data, savepath='.'):
        '''
            Extract the abstracts from the xml search query response and download the pdf paper and extract the plain text from it and remove possible abstract in there

            data (str): xml data with all paper urls and abstracts
            savepath (str): save path for the txt files
        '''
        # build xml tree
        root = etree.fromstring(data)
        # reserve lists
        abstracts = []
        papers = []
        # extract abstract directly from summary tag and extract pdf url
        for child in root:
            if len(child) > 0 and child.tag == '{http://www.w3.org/2005/Atom}entry':
                for grandchild in child:
                    if grandchild.tag == '{http://www.w3.org/2005/Atom}summary':
                            abstracts.append(grandchild.text)
                    if grandchild.tag == '{http://www.w3.org/2005/Atom}link' and 'title' in grandchild.attrib and grandchild.attrib['title'] == 'pdf':
                            papers.append(grandchild.attrib['href'])
        
        ## download pdfs, extract plain text, remove possible abstracts in there
        for i, paper in enumerate(papers):
            # download pdf
            pdf = urllib.request.urlopen(paper).read()
            # save pdf temporarily as file
            filename = paper.split('/')[-1]
            with open(os.path.join(savepath,'temp', filename + '.pdf'), 'wb+') as f:
                    f.write(pdf)
            # parse pdf file to get psdf text content and replace url with text content
            parsed = parser.from_file(os.path.join(savepath,'temp', filename + '.pdf'))['content']
            # remove line breaks with hyphenation in paper and abstract
            hyph_norm_parsed_paper = re.sub('-\n', '', parsed)
            abstract_rgex_pattern = re.sub('-\n', '', abstracts[i])
            # make list of words from abstract
            abstract_words = abstract_rgex_pattern.split()
            # remove abstract by searching for paragraph starting with abstract and ending with double whitespace
            parsed = re.sub('(?i)(\s*)(abstract)([\s\S]' + '{0,' + str(len(abstract_rgex_pattern)*4//3) + '})' + re.escape(abstract_rgex_pattern[-3:-1]) + '\s\s', '\n', hyph_norm_parsed_paper)
            # remove abstract by searching for paragraph start and end of abstract
            parsed = re.sub('(?i)(\s*)' + re.escape(abstract_rgex_pattern[0:3]) + '([\s\S]' + '{0,' + str(len(abstract_rgex_pattern)*4//3) + '})' + re.escape(abstract_rgex_pattern[-3:-1]) , '\n', parsed)
            # remove abstract by searching for paragraph start word and end word of abstract
            parsed = re.sub('(?i)(\s*)' + re.escape(abstract_words[0]) + '([\s\S]' + '{0,' + str(len(abstract_rgex_pattern)*4//3) + '})' + re.escape(abstract_words[-1]), '\n', parsed)
            # remove abstract heading in case it was not found before
            parsed = re.sub('(?i)(\s*)(abstract)', '\n', parsed)
            # remove unnecessary whitespace
            abstracts[i] = ' '.join(abstract_words)
            parsed = ' '.join(parsed.split())
            papers[i] = parsed

        # remove temporary pdfs
        shutil.rmtree(os.path.join(savepath,'temp'))
        return abstracts, papers