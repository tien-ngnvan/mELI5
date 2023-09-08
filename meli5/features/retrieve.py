import os
import logging
import torch
from typing import Dict, List, Tuple

import psycopg2
from psycopg2 import Error

from transformers import PreTrainedModel
from datasets import Dataset 

from .base import BaseDataProcess
        
logger = logging.getLogger(__name__)


class RetrieveELI5(BaseDataProcess):
    def __init__(
        self,
        model:PreTrainedModel,
        tokenizer,
        max_length:int=None,
        num_proc:int=None,
        db_connector:Dict=None,
        device:str=None
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.db_conn = db_connector
        self.num_proc = num_proc
        self.device = device
        
    
    def query(self, embds, tbname:str=None, threshold:int=27, limit:int=10) -> List[Tuple]:
        """ Query database, take input is string array embebding and 
        calculated similarity by pgvector database.

        Args:
            embds (_type_): string array embedding
            tbname (str, optional): table name query. Defaults to None.
            threshold (int, optional): number of threshold similarity, 
                higher is quality documents. Defaults to 27.
            limit (int, optional): number of return passage documents. Defaults to 10.

        Returns:
            List[Tuple]: a result query from database

        """
        
        try:
            conn = psycopg2.connect(**self.db_conn)
            conn.autocommit = True
            cursor = conn.cursor()
            
            sql = f'''
                    SET LOCAL ivfflat.probes = 3;
                    SELECT
                        tb.id,
                        tb.content,
                        (tb.embedd <#> %s) * -1 AS score
                    FROM {tbname} as tb
                    WHERE (tb.embedd <#> %s) * -1 > %s
                    ORDER BY score DESC
                    LIMIT %s
                '''  
            cursor.execute(sql, (embds, embds, threshold, limit))
            results = cursor.fetchall() 
            
            if len(results) < limit:
                logger.info("No query results or less than limit samples. Set query default setting")
                sql = f'''
                    SELECT
                        tb.id,
                        tb.content,
                        (tb.embedd <#> %s) * -1 AS score
                    FROM {tbname} as tb
                    ORDER BY score DESC
                    LIMIT %s
                '''
                cursor.execute(sql, (embds, limit))
                results = cursor.fetchall()
                
            if conn:
                cursor.close
                conn.close
                
            return results
        
        except(Exception, Error) as error:
            logging.info(f'PostgreSQL: {error}')
        
        
    def convert_embds(self, examples:Dataset) -> Dict[str, str]:
        """ Convert string input to embedding

        Args:
            examples (Dataset): a single or batch samples text

        Returns:
            Dict[str, str]: embedding is converted to string array
        """
        
        tokenized = self.tokenizer(examples, padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors='pt')
        
        with torch.no_grad():
            output = self.model(**tokenized.to(self.device))["pooler_output"]
            
        # run with batch_size
        if len(examples) > 1:
            embd = [
                str(list(output[idx,:].cpu().detach().numpy().reshape(-1))) 
                for idx in range(output.size(0))
            ]
        else:
            embd = str(list(output.cpu().detach().numpy().reshape(-1)))
            
        return {'embds' : embd}
    
    
    def process(self, example:Dataset, tbname:str=None) -> Dict[str, List[str]]:
        """ Processing each sentence 

        Args:
            example (Dataset): a single example from dataset
            tbname (str, optional): table query from database. Defaults to None.

        Returns:
            - passages: results by query function
            - retrieve_score: a similarity score between query and passages return by database
            
        """
        
        result_query = self.query(example['embds'], tbname)
        output = {
            'passages':[str(item[0]) + ' ## ' + item[1].strip() for item in result_query],
            'retrieve_score':[str(item[0]) + ' ## ' + str(round(item[2],1)) for item in result_query]
        }
        
        return output
    
    
    def save(self, datasets:Dataset, output_path:str):
        """ Save datasets

        Args:
            datasets (Dataset): huggingface datasets
            output_path (str): save output dir
        """
        
        datasets.to_json(output_path)
        logger.info(f'Saved datasets at {output_path}')
    
    
    def run(self, datasets:Dataset, tbname:str, output_path:str=None) -> Dataset:
        """ Main processing retrieve datasets

        Args:
            datasets (Dataset): datasets huggingface hub
            output_path (str, optional): output save datasets. Defaults to None.
            tbname (str, optional): table query database

        Returns:
            Datasets: processed datasets
        """
        
        num_worker = os.cpu_count() if self.num_proc > os.cpu_count() else self.num_proc
        
        # First get embedding query by batch
        datasets = datasets.map(
            lambda examples : self.convert_embds(examples['en_inp']),
            batched=True, batch_size=256,
            desc="Tokenized query sentence process"
        )       
        
        # Open multi-process query database
        datasets = datasets.map(
            lambda example : self.process(example, tbname),
            remove_columns='embds',
            desc= 'Query database process'
        ) 
        
        # save datasets
        if output_path:
            self.save(datasets, output_path)
        
        return datasets