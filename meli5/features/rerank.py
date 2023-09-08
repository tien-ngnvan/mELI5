import re
import torch
import logging
from typing import List, Dict

from symsearch import RerankInference
from transformers import PreTrainedModel
from datasets import Dataset
from .base import BaseDataProcess

logger = logging.getLogger(__name__)


class BatchRerankingInference(RerankInference):
    def __init__(self, model_name_or_path: str = None, 
                 q_max_length: int = 96, p_max_length: int = 384, 
                 device_type: str = 'cpu') -> None:
        super().__init__(model_name_or_path, q_max_length, p_max_length, device_type)
        
    def encode_batch(self, queries:str, passages:List[str]) -> List[float]:
        """ Rerank input pairs sentence by calculated inner product  
        between queries and passages.

        Args:
            queries (str, optional): a query is a question
            passages (Listr[str], optional): a list of passages 

        Returns:
            List[float]: List of score similarity
        """
        
        if isinstance(queries, str):
            num_pg = len(passages)
            queries = [queries] * num_pg
            
        assert len(queries) == len(passages), f'Number of queries not the same number of passges'
        
        tokenized = self.tokenizer(
            queries, passages,
            max_length= self.q_max_length + self.p_max_length,
            padding=True, truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            model_output = self.model(pair=tokenized)
            scores = model_output.scores.cpu().detach().numpy()
        
        #  multiply by -1 since score return negative iner product
        return [item[0]*-1 for item in scores]
    
    
class RerankELI5(BaseDataProcess):
    def __init__(
        self, 
        model:PreTrainedModel,
        max_length:int=None,
        num_proc:int=None,
        device:str=None,
        batch_szie:int=None
        ):
        self.model = model
        self.max_length = max_length
        self.num_proc = num_proc
        self.device = device
        self.batch_size = batch_szie
        
    def split_text(self, sentence:str=None):
        """ split input sentence into id and text

        Args:
            sentence (str, optional): a single sentence. Defaults to None.

        Returns:
            id: id of sentence
            text: a sentence
        """
        
        assert isinstance(sentence, str), f"Input should be a string not {type(sentence)}"
        id, text = re.split(' ## ', sentence, 1)

        return id, text
    
    def process(self, examples:Dataset, model:PreTrainedModel) -> Dict[str, List[str]]:
        """ Processing input examples

        Args:
            examples (Dataset): a batch or single example
            model (PreTrainedModel): a PreTrainModel

        Returns:
            Dict[str, List[str]]: a PreTrainModel
        """
        
        lsa = []
        for _, (q,p) in enumerate(zip(examples['en_inp'], examples['passages'])):
            ls_id, ls_pg = [], []
            for sentence in p:
                id_sample, text = self.split_text(sentence)
                ls_id.append(id_sample)
                ls_pg.append(text)
                
            result = model.encode_batch(q, ls_pg)
            result = list(zip(ls_id, result))
            sorted_result = sorted(result, key=lambda item:item[1], reverse=True)

            lsa.append(
                [str(item[0]) + ' ## ' + str(round(item[1], 1)) for item in sorted_result]
            )
        
        return {'rerank_score' : lsa}
    
    def save(self, datasets, output_path):
        """ Save datasets

        Args:
            datasets (Dataset): huggingface datasets
            output_path (str): save output dir
        """
        
        datasets.to_json(output_path)
        logger.info(f'Saved datasets at {output_path}')
        
    def run(self, datasets, output_path:str=None) -> Dataset:
        """ Main processing clean datasets

        Args:
            datasets (Dataset): datasets huggingface hub
            output_path (str, optional): output save datasets. Defaults to None.

        Returns:
            Datasets: processed datasets
        """
        
        # main processing Rerank sample
        datasets = datasets.map(
            lambda examples: self.process(examples, self.model),
            batched=True, batch_size=self.batch_size,
            desc='Rerank sample process'
        )
        
        # save datasets
        if output_path:
            self.save(datasets, output_path)
        
        return datasets
    
    