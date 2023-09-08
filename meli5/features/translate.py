import re
import torch
import logging
from .base import BaseDataProcess
from typing import Union, List, Dict

from transformers import PreTrainedModel
from datasets import Dataset

logger = logging.getLogger(__name__)


class TranslateELI5(BaseDataProcess):
    def __init__(
        self,
        model:PreTrainedModel,
        tokenizer,
        device:str = None,
        batch_size:int=None
        ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        
    def translate(self, examples) -> Union[str, List[str]]:
        """ Translate function take input sentence or batch sentence translate english
        to Vietnamese datasets. 

        Args:
            examples (Union[str, List[str]]): _description_

        Returns:
            Union[str, List[str]]: _description_
        """
        
        tokenized = self.tokenizer(examples, padding='longest', truncation=True,
                                return_tensors='pt').input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(tokenized, max_length=1024)
            results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return results
        
    def process(self, examples:Dataset, key:str=None) -> Dict[str, List[str]]:
        """Processing each sentence 

        Args:
            examples (Dataset): is single sentence or batch sentence input
            key (_type_): key is a train/val/test. If datasets file is a test, with ELI5 no answer/label

        Returns:
            Dict[str, List[str]]: _description_
        """
        
        # translate query
        query_prompt = ['en: ' + example for example in examples['en_inp']]
        query_translated = self.translate(query_prompt)
        
        # translate passages
        pgs_translated = []
        for example in examples['passages']:
            passages, id_ls, result = [], [], []
            # get only text to translate ignore id_number
            for p in example:
                id_num, text = re.split(' ## ', p , 1)
                passages.append('en: ' + text)
                id_ls.append(id_num)

            # concat to original sample
            model_output = self.translate(passages)
            for _, (idx, inp_translated) in enumerate(zip(id_ls, model_output)):
                out = idx + " ## " + inp_translated[3:].strip()
                result.append(out)
            pgs_translated.append(result)
        
        # translate answer
        if key != 'test':
   
            answer_prompt = ['en: ' + example for example in examples['en_opt']]
            answer_translated = self.translate(answer_prompt)
        
            return {
                'vi_inp' : [line[3:] for line in query_translated],
                'vi_opt' : [line[3:] for line in answer_translated],
                'vi_passages' : pgs_translated
            }
            
        return {
            'vi_inp' : [line[3:] for line in query_translated],
            'vi_passages' : pgs_translated
        }
    
    def save(self, datasets:Dataset, output_path:str):
        """ Save datasets

        Args:
            datasets (Dataset): huggingface datasets
            output_path (str): save output dir
        """
        
        datasets.to_json(output_path, force_ascii=False)
        logger.info(f'Saved datasets at {output_path}')
    
    def run(self, datasets:Dataset, key:str=None, output_path:str=None) -> Dataset:
        """ Main processing translate datasets

        Args:
            datasets (Dataset): datasets huggingface hub
            key (str, optional): key is a train/val/test. If datasets file is a test, with ELI5 no answer/label. Defaults to None.
            output_path (str, optional): output save datasets. Defaults to None.

        Returns:
            Datasets: processed datasets
        """
        datasets = datasets.map(
                lambda examples : self.process(examples, key), 
                batched=True, batch_size=self.batch_size,
                desc='Translate English to Vietnamese'
        )
        
        # save datasets
        if output_path:
            self.save(datasets, output_path)
        
        return datasets
        