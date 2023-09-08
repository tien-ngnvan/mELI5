import re
import os
import logging
from unstructured.cleaners.core import (
    replace_unicode_quotes,
    clean_non_ascii_chars,
    clean_extra_whitespace,
)
from unstructured.partition.html import partition_html
from datasets import Dataset
from .base import BaseDataProcess

logger = logging.getLogger(__name__)


class CleanText(BaseDataProcess):
    def __init__(
        self,
        replace_unicode:bool=False,
        clean_non_ascii:bool=False,
        clean_whitespace:bool=False,
        clean_html:bool=False,
        remove_special_token=False,
        num_proc:int=4
    ) -> None:
        self.replace_unicode = replace_unicode
        self.clean_non_ascii = clean_non_ascii
        self.clean_whitespace = clean_whitespace
        self.remove_special_token = remove_special_token
        self.clean_html = clean_html
        self.num_proc = num_proc

    def clean_fn(self, text:str) -> str:
        """
        Clean text:
        -----------------------
        replace_unicode: Whether to unicode characters in text.
        clean_non_ascii: Whether to clean dash characters in text.
        clean_whitespace: Whether to clean multiple whitespace bullets from a section of text.
        clean_html: Whether to clean html tags in text.
        remove_special_token: Whether to remove '\n' in text.

        """
        assert isinstance(text, str), f'Input should be a string not {type(text)}'
        
        if self.remove_special_token:
            cleaned_text = re.sub('\n', ' ', text)
            cleaned_text = re.sub('\_+URL\_+[0-9]', '', text)

        cleaned_text = replace_unicode_quotes(text) if self.replace_unicode else cleaned_text
        cleaned_text = clean_non_ascii_chars(cleaned_text) if self.clean_non_ascii else cleaned_text

        if self.clean_html:
            elements = partition_html(text=cleaned_text)
            cleaned_text = ' '.join([element.text for element in elements])

        cleaned_text = clean_extra_whitespace(cleaned_text) if self.clean_whitespace else cleaned_text

        return cleaned_text.strip()
    
    def process(self, example:Dataset, key:str=None) -> Dataset:
        """ Processing each sentence 

        Args:
            example (Dataset): a single example from dataset
            key (str, optional): key is a train/val/test. If datasets file is a test, 
                with ELI5 no answer/label. Defaults to None.

        Returns:
            Dataset: processed dataset
        """
        
        if key != 'test':
            example['en_opt'] = self.clean_fn(example['en_opt'])
        
        example['en_inp'] = self.clean_fn(example['en_inp'])
        example['passages'] = [self.clean_fn(ex) for ex in example['passages']]
        
        return example
    
    def save(self, datasets:Dataset, output_path:str):
        """ Save datasets

        Args:
            datasets (Dataset): huggingface datasets
            output_path (str): save output dir
        """
        
        datasets.to_json(output_path)
        logger.info(f'Saved datasets at {output_path}')
    
    def run(self, datasets, key:str='train', output_path:str=None) -> Dataset:
        """ Main processing clean datasets

        Args:
            datasets (Dataset): datasets huggingface hub
            key (str, optional): key is a train/val/test. If datasets file is a test, 
                with ELI5 no answer/label. Defaults to None.
            output_path (str, optional): output save datasets. Defaults to None.

        Returns:
            Datasets: processed datasets
        """
        
        num_worker = os.cpu_count() if self.num_proc > os.cpu_count() else self.num_proc
        
        datasets = datasets.map(
            lambda example: self.process(example, key),
            desc='Clean text process',
            num_proc= num_worker
        )
        
        # save datasets
        if output_path:
            self.save(datasets, output_path)
            
        return datasets