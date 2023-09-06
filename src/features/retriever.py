import torch
import os
import argparse
import glob
from typing import Dict, List, Tuple

import psycopg2
from psycopg2 import Error

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from datasets import Dataset, load_dataset


DICT_PST = {
    'dbname':'',
    'host':'',
    'port':'',
    'user':'',
    'password':''
}


def get_passage(embedding:str, tbname, threshold=25, num_doc=10) -> List[Tuple]:
    """Query passage from postgresql 

    Args:
        embedding (str, optional): is a list of numpy vectors
        tbname (str, optional): table from database from 
        threshold (int, optional): An appropriate threshold filter low quality documents. Defaults to 25.
        num_doc (int, optional): number of matches documents to query. Defaults to 10.

    Returns:
        List of tuple result query from database
    """
    try:
        conn = psycopg2.connect(**DICT_PST)
        conn.autocommit = True
        cursor = conn.cursor()

        sql = f'''
            SET LOCAL ivfflat.probes = 3;
            SELECT
                tb.id,
                tb.content,
                (tb.embedd <#> %s) * -1 AS score
            FROM {tbname} as tb
            WHERE (tb.embedd <#> %s) * -1 > {threshold}
            ORDER BY score DESC
            LIMIT {num_doc}
        '''

        cursor.execute(sql, (embedding, embedding))
        results = cursor.fetchall()

        if conn:
            cursor.close()
            conn.close()

        return results

    except(Exception, Error) as error:
        print(f'Error: {error}')


def tokenized_fn(examples:Dataset, model, tokenizer, device:str='cpu') -> Dict[str, List[str]]:
    """ Tokenize function 

    Args:
        examples: batch dataset from Huggingface Dataset
        model: model load from Huggingface hub
        tokenizer: A tokenizer encode sentence to token ids
        device (str):  Device (like 'cuda' / 'cpu') that should be used for computation

    Returns:
        Dictionary of embedding 
    """
    tokenized = tokenizer(examples, padding=True, truncation=True,
                          max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        model_outputs = model(**tokenized)["pooler_output"]

    embds = [str(list(model_outputs[i, :].cpu().detach().numpy().reshape(-1)))
                    for i in range(model_outputs.size(0))]

    return {'embds':embds}


def process_fn(example:Dataset, tbname:str=None) -> Dataset:
    """ Process output query into format dataset huggingface

    Args:
        example (_type_): _description_
        tbname (_type_): _description_

    Returns:
        _type_: _description_
    """
    results = get_passage(example, tbname)
    output  = {
        'passages': [str(item[0]) + ' ## ' + item[1].strip() for item in results],
        'retrieve_score':[str(item[0]) + ' ## ' + str(round(item[2],1)) for item in results]
    }
    return output


def main(args):
    # update postgres connector
    DICT_PST.update({
        'dbname': args.postgres_dbname,
        'host': args.postgres_host,
        'port': args.postgres_port,
        'user': args.postgres_user,
        'password': args.postgres_pw
    })
    
    # setup model & tokenizer
    device = torch.device("cuda:0")
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.model_name_or_path)
    model = DPRQuestionEncoder.from_pretrained(args.model_name_or_path).to(device)

    # Use in case chunk large dataset into sub-dataset
    if args.partition: 
        extention = args.data_name_or_path.split('.')[-1]
        datasets = load_dataset(extention, data_files=args.data_name_or_path, split='train')
        datasets.save_to_disk('./data/raw', num_shards=20)

        del datasets

    # load sub-dataset
    for name in glob.glob('./data/raw/*.arrow'):
        path_name = name.split('/')[-1]
        
        print(f'\n================== Processing {path_name} file =========================')
        datasets = load_dataset('arrow', data_files=name, split='train')

        datasets = datasets.map(
            lambda example:tokenized_fn(example['en_inp'], model, tokenizer, device),
            batched=True, batch_size=64,
            desc='Get all embedding samples',
        )

        # open multi processes query make fast 
        datasets = datasets.map(
            lambda example:process_fn(example['embds'],args.postgres_tbname),
            desc='Query documents from database',
            remove_columns='embds',
            num_proc=os.cpu_count() if args.num_proc > os.cpu_count() else args.num_proc,
        )
        
        datasets.to_json(f"./data/interim/{path_name.split('.')[-1]}.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generative QA")
    parser.add_argument('--postgres_host', type=str,
                        default='dev.gradients.host', help="PostgreSQL")
    parser.add_argument('--postgres_port', type=str,
                        default='5432', help="PostgreSQL port connect database")
    parser.add_argument('--postgres_user', type=str,
                        default='postgres', help="User connect database")
    parser.add_argument('--postgres_pw', type=str,
                        default='*********', help="Password connect database")
    parser.add_argument('--postgres_dbname', type=str,
                        default='wikidatabase', help="Database in postgreSQL")
    parser.add_argument('--postgres_tbname', type=str,
                        default='wiki_tb_17m_128', help="Table in postgreSQL")
    parser.add_argument('--model_name_or_path', type=str,
                    default='vblagoje/dpr-question_encoder-single-lfqa-wiki', help="Query encoder model")
    parser.add_argument('--data_name_or_path', type=str,
                        default='/data.json', help="Data folders")
    parser.add_argument('--num_proc', type=int,
                        default=4, help="Number of process worker")
    parser.add_argument('--partition', type=bool,
                        default=False, help="Partition large dataset into small dataset")
    # prepare args
    args = parser.parse_args()

    main(args)