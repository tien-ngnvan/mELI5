import torch
import glob
import argparse
import os
from typing import List, Dict

from datasets import load_dataset, Dataset
from symsearch import RerankInference
from transformers import PreTrainedModel


class BachReranking(RerankInference):
    def __init__(self, model_name_or_path: str = None, 
                 q_max_length: int = 96, 
                 p_max_length: int = 384, 
                 device_type: str = 'cpu') -> None:
        super().__init__(model_name_or_path, q_max_length, p_max_length, device_type)
        
    def encode_batch(self, queries:List[str], passages:List[str]):
        """_summary_

        Args:
            queries (List[str]): _description_
            passages (List[str]): _description_

        Returns:
            _type_: _description_
        """
        if isinstance(queries, str):
            num_pg = len(passages)
            queries = [queries] * num_pg

        assert len(queries) == len(passages), f" Number of queries not the same number of passages"

        tokenized = self.tokenizer(queries, passages,
                                   max_length= self.q_max_length + self.p_max_length,
                                   truncation='only_first', padding='max_length',
                                   return_attention_mask=True, return_token_type_ids=True,
                                   return_tensors='pt').to(self.device)

        model_output = self.model(pair=tokenized)
        scores = model_output.scores.cpu().detach().numpy()
        result = [item[0] * -1 for item in scores]

        return result
    
    
def split_text(sentence:str=None):
    """_summary_

    Args:
        sentence (str, optional): _description_. Defaults to None.

    Returns:
        id: Id of example
        text: passages containt
    """
    assert isinstance(sentence, str), f"Input should be a string not {type(sentence)}"
    id, text = sentence.split(' ## ')

    return id, text


def process_fn(examples:Dataset=None, model:PreTrainedModel=None) -> Dict[str, List]:
    """ Process scoring sample
    Args:
        examples (Dataset, optional): Batch example in process function. 
        model (PreTrainedModel, optional): Pretrain model load from HF.

    Returns:
       Dataset:  new dataset with append rerank_score column
    """
    lsa = []
    for _, (q, p) in enumerate(zip(examples['en_inp'], examples['passages'])):
        ls_id, ls_pg = [], []
        for item in p:
            id, text = split_text(item)
            ls_id.append(id)
            ls_pg.append(text)

        result = model.encode_batch(q, ls_pg)
        result = list(zip(ls_id, result))

        sorted_by_score = sorted(result, key=lambda tup: tup[1], reverse=True)
        lsa.append([str(item[0]) + ' ## ' + str(round(item[1],1)) for item in sorted_by_score])

    return {'rerank_score': lsa}


def main(args):
    # setup model & tokenizer
    #device = torch.device("cpu")
    model = BachReranking(args.model_name_or_path, device_type='gpu')
    
    # load sub-dataset
    for name in glob.glob(os.path.join(args.data_path, '*.json')):
        path_name = name.split('/')[-1]
        print(f'\n================== Processing {path_name} file =========================\n')
        
        dataset = load_dataset('json', data_files=name, split='train').select(range(20))
        dataset = dataset.map(
            lambda examples : process_fn(examples, model), 
            batched=True, batch_size=args.batch_size,
            desc="Rerank samples process")
        
        dataset.to_json(os.path.join(args.output, path_name))
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generative QA")
    parser.add_argument('--model_name_or_path', type=str,
                    default='caskcsg/cotmae_base_msmarco_reranker', help="Rerank model")
    parser.add_argument('--data_path', type=str,
                        default='/data.json', help="Data folders")
    parser.add_argument('--output', type=str,
                        default='./data/processed/train', help="Data folders")
    parser.add_argument('--batch_size', type=int,
                        default=4, help="Batch process datasets")
    
    args = parser.parse_args()

    main(args)