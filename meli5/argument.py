import argparse
import sys

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,
                        default=4, help="Batch process datasets")
    parser.add_argument('--device_type', type=str,
                        default='cpu', help="Model process data")
    parser.add_argument('--data_name_or_path', type=str,
                        default='/data.json', help="Data folders")
    parser.add_argument('--output', type=str,
                        default='./data/processed/', help="Data folders")
    parser.add_argument('--max_length', type=int,
                        default=512, help="Max token length samples")
    parser.add_argument('--num_proc', type=int,
                        default=4, help='Number of processes')

    # For retrieval
    parser.add_argument('--model_name_or_path_retrieve', type=str,
                    default='vblagoje/dpr-question_encoder-single-lfqa-wiki', help="Retrieve model")
    parser.add_argument('--max_length_retrieve', type=int,
                        default=96, help="Max token length samples")
    parser.add_argument('--postgres_host', type=str,
                        default='dev.gradients.host', help="PostgreSQL")
    parser.add_argument('--postgres_port', type=int,
                        default=5432, help="PostgreSQL port connect database")
    parser.add_argument('--postgres_user', type=str,
                        default='postgres', help="User connect database")
    parser.add_argument('--postgres_pw', type=str,
                        default='UgTL3ZHZic7J', help="Password connect database")
    parser.add_argument('--postgres_dbname', type=str,
                        default='wikidatabase', help="Database in postgreSQL")
    parser.add_argument('--postgres_tbname', type=str,
                        default='wiki_tb_17m_128', help="Table in postgreSQL")

    # For reranker
    parser.add_argument('--model_name_or_path_reranker', type=str,
                    default='caskcsg/cotmae_base_msmarco_reranker', help="Rerank model")

    # For clean text
    parser.add_argument('--replace_unicode', type=bool,
                        default=True, help='remove unicode characters in text')
    parser.add_argument('--clean_non_ascii', type=bool,
                        default=True, help='clean dash characters in text')
    parser.add_argument('--clean_whitespace', type=bool,
                        default=True, help='remove multiplespace in text')
    parser.add_argument('--clean_html', type=bool,
                        default=True, help='remove html tags')
    parser.add_argument('--remove_special_token', type=bool,
                        default=True, help="remove multiple '\n' | '\t' in text")

    # For translate
    parser.add_argument('--model_name_or_path_translation', type=str,
                    default='VietAI/envit5-translation', help="Translate model")
    parser.add_argument('--batch_size_translation', type=int,
                    default=2, help="Batch process datasets")


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
        
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    
    return args