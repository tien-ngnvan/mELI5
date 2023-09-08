import torch
import os
import logging
import glob

from features import (
    RetrieveELI5,
    RerankELI5,
    BatchRerankingInference,
    CleanText,
    TranslateELI5
)
# from features.retrieve import RetrieveELI5
# from features.rerank import RerankELI5, BatchRerankingInference
# from features.cleantext import CleanText
# from features.translate import TranslateELI5

from datasets import load_dataset
from transformers import (
    DPRQuestionEncoder, 
    DPRQuestionEncoderTokenizer,
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
)

from argument import arguments 

logger = logging.getLogger(__name__)

def main(args):
    # setup postgres connector
    DICT_PST = {
        'dbname': args.postgres_dbname,
        'host': args.postgres_host,
        'port': args.postgres_port,
        'user': args.postgres_user,
        'password': args.postgres_pw
    }
    # setup device
    if torch.cuda.device_count() >= 2:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
    elif torch.cuda.device_count() == 1:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
    
    ################## setup models and tokenizers Retrieve ##################
    model_retrieve = DPRQuestionEncoder.from_pretrained(args.model_name_or_path_retrieve)
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.model_name_or_path_retrieve)
    retrieveELI5 = RetrieveELI5(
        model=model_retrieve.to(device0), 
        tokenizer=tokenizer,
        max_length=args.max_length_retrieve,
        num_proc=args.num_proc,
        db_connector= DICT_PST,
        device=device0
    )
    
    ################## setup models and tokenizers Rerank ##################
    model_ranking = BatchRerankingInference(args.model_name_or_path_reranker)
    rerankELI5 = RerankELI5(
        model=model_ranking,
        max_length=args.max_length,
        num_proc=args.num_proc,
        device=device0,
        batch_szie=args.batch_size
    )
    
    ################## setup CleanText ##################
    clean = CleanText(replace_unicode=True, clean_non_ascii=True,
                      clean_whitespace=True, clean_html=True,
                      remove_special_token=True,num_proc=args.num_proc)
    
    ################## setup Translation ##################
    model_translate = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path_translation)
    tokenizer_translate = AutoTokenizer.from_pretrained(args.model_name_or_path_translation)
    translateELI5 = TranslateELI5(
        model=model_translate.to(device1),
        tokenizer=tokenizer_translate,
        device=device1,
        batch_size=args.batch_size_translation
    )
    
    ################## Build datasets ##################
    print('\n\n', '#'*100, '\n')
    if not os.path.exists(args.data_name_or_path):
        datasets = load_dataset(args.data_name_or_path)
        if len(datasets) > 1e+5:
            logging.info(
                'Processing take long time to query database. We suggest load sub-dataset'
                )   
        print('Datasets info:\n', datasets[0])
        
        output_path = os.path.join(args.output, f'{args.data_name_or_path}.json')
        datasets = retrieveELI5.run(datasets, tbname=args.postgres_tbname)
        datasets = rerankELI5.run(datasets)
        datasets = clean.run(datasets)
        datasets = translateELI5.run(datasets, output_path=output_path)
            
    elif os.path.exists(args.data_name_or_path):
        for path_name in glob.glob(os.path.join(args.data_name_or_path, '*')):
            name_file = path_name.split("\\")[-1]
            key = path_name.split('\\')[-2]

            print(f'\n\n================== Processing {name_file} file =========================')
            
            extention = name_file.split('.')[-1]
            try:
                datasets = load_dataset(extention, data_files=path_name, split='train').select(range(4))
            except:
                logger.infor(f'Error loading file {path_name}')
                
            print('Datasets info:\n', datasets)

            output_path = os.path.join(args.output, name_file)
            datasets = retrieveELI5.run(datasets, tbname=args.postgres_tbname)
            datasets = clean.run(datasets, key)
            datasets = rerankELI5.run(datasets)
            datasets = translateELI5.run(datasets, key, output_path=output_path)
    else:
        logger.info('\n\nDo not recognize data_name_or_path\n\n')

if __name__ == '__main__':
    args = arguments()
    main(args)

