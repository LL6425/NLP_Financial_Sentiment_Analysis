from typing import List

class TextCleaner:

    def __init__(self,
                 spacy_vocab : str = 'en_core_web_sm',
                 stopwords_adjustments: dict = {},
                 remove_stopwords: bool = True,
                 remove_numbers: bool = True,
                 remove_recognized_entities: bool = True,
                 remove_punct: bool = True,
                 word_red:str = 'stem'):
        
        import spacy
        from spacy.lang.en import STOP_WORDS as eng_stop_words
        
        self.nlp = spacy.load(spacy_vocab)
        self.stopwords = eng_stop_words.union(set(
            stopwords_adjustments['+'] if '+' in stopwords_adjustments.keys() else [])) - set(
                stopwords_adjustments['-'] if '-' in stopwords_adjustments.keys() else [])
        self.removing = {'stopwords': remove_stopwords, 
                         'numbers': remove_numbers, 
                         'entities': remove_recognized_entities, 
                         'punct': remove_punct}
        self.word_red = word_red

        if self.word_red=='stem':

            from nltk.stem.porter import PorterStemmer

            self.stemmer = PorterStemmer()

        self.check_input(spacy_vocab, stopwords_adjustments, self.removing, word_red)

        

    def check_input(self, 
                    vocab: str, 
                    stopwords_adjustments: dict, 
                    removing_dict: dict, 
                    word_red: str) -> None:

        admitted_vocab_str = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg']
        
        assert vocab in admitted_vocab_str, f'Vocabulary string {vocab} not admitted. Managed are {admitted_vocab_str}'
        
        admitted_keys = ['+','-']
        
        assert set(stopwords_adjustments.keys()).issubset(set(admitted_keys)), f'Stopwords append/removal failed due to non admitted keys. Managed keys are {admitted_keys}'

        removing_dict_check_dtypes = [isinstance(cond, bool) for cond in removing_dict.values()]

        assert len(set(removing_dict_check_dtypes))==1 and removing_dict_check_dtypes[0], f'Only boolean admitted'

        admitted_word_reductions = ['stem', 'lem', None]

        assert word_red in admitted_word_reductions, f'{word_red} not admitted. Managed are {admitted_word_reductions}'


    def clean_single_str(self, hl: str) -> str:

        doc = self.nlp(hl)

        to_remove = list(self.stopwords) if self.removing['stopwords'] else []

        to_remove += [token.text for token in doc if token.is_digit] if self.removing['numbers'] else []
        to_remove += [token.text for token in doc if token.is_punct] if self.removing['punct'] else []

        entities = [e.text for e in doc.ents] if self.removing['entities'] else []

        if self.word_red=='stem':

            clean_l = [self.stemmer.stem(token.text.lower()) for token in doc if token.text.lower() not in to_remove and token.text not in entities]


        elif self.word_red=='lem':


            clean_l = [token.lemma_.lower() for token in doc if token.text.lower() not in to_remove and token.text not in entities]

        else:

            clean_l = [token.text.lower() for token in doc if token.text.lower() not in to_remove and token.text not in entities]


        return ' '.join(clean_l)
    
    
    def clean_ls(self, ls: List[str]) -> str:

        cleaned_strings = [self.clean_single_str(single_str) for single_str in ls]

        return ' '.join(cleaned_strings)
    