import datasets
from datasets import load_dataset
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import warnings
import numpy as np
from pathlib import Path

models = {"small":"ai4bharat/indictrans2-en-indic-dist-200M",
          "large": "ai4bharat/indictrans2-en-indic-1B"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ip = IndicProcessor(inference=True)


def initialize_model_and_tokenizer(ckpt_dir, quantization):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()

    return tokenizer, model



def get_preds(batch,tokenizer,model):

    inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=512,
            num_beams=5,
            num_return_sequences=1,
        )
    # Decode the generated tokens into text
            
    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
    return generated_tokens

        # Postprocess the translations, including entity replacement
        

def preprocess_texts(examples,src_lang='eng_Latn',tgt_langs=[],tokenizer=None,model=None):
    
    for tgt_lang in tgt_langs:
        #first preprocess the questions
        batch = ip.preprocess_batch(
                examples['question'],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
        
        generated_tokens = get_preds(batch,tokenizer,model)
        translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        examples[f'question_{tgt_lang}'] = translations
        
         
        #convert the options.
        n_options = len(examples['options'][0])
        flat_options = np.array(examples['options']).flatten()
        
        batch = ip.preprocess_batch(
                flat_options,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
        
        generated_tokens = get_preds(batch,tokenizer,model)
        translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
        
        translations = np.array(translations).reshape(-1,n_options).tolist()
        
        examples[f'options_{tgt_lang}'] = translations
        
        return examples





def process_dataset(ds,output_dir,tokenizer, model, src_lang='eng_Latn',target_langs=['hin_Deva'],):
    cols = set(ds.column_names) - {'question','options'}
    cols = list(cols)
    translated_ds = ds.map(preprocess_texts, batched=True, fn_kwargs={"src_lang":src_lang, 
                                                                      'tgt_langs':target_langs,
                                                                      "tokenizer":tokenizer,
                                                                      "model":model,
                                                                      })
    for tgt_lang in target_langs:
        lang_ds = translated_ds.select_columns(cols+[f'question_{tgt_lang}',f'options_{tgt_lang}'])
        lang_ds = lang_ds.rename_columns({
            f'question_{tgt_lang}':'question',
            f'options_{tgt_lang}':'options'
        })
        
        lang_ds.save_to_disk(output_dir/f'{tgt_lang}')
    
   
    

def main(args):
    get_mmlu = args.dataset in {'all','mmlu'}
    get_mmlu_pro = args.dataset in {'all','mmlupro'}
    print(args)
    print(get_mmlu, get_mmlu_pro)
    
    model_name = models[args.size]
    
    OUT_DIR = Path(args.output)
    
    quantization = f"{args.quantization}-bit" if args.quantization else None
    
    tokenizer, model = initialize_model_and_tokenizer(model_name,quantization)
    process_ds_args = {
        "src_lang":"eng_Latn",
        "target_langs":args.target_langs,
        "tokenizer":tokenizer,
        "model":model
    }
    
    print((f"""\n\n Processing {args.dataset} dataset(s)
            Using Device :{DEVICE}"
            Model :{model_name}
            Quantization :{quantization}
            Target Languages :{args.target_langs}
          """))
    
    datasets_to_process=[]
    if get_mmlu:
        ds = load_dataset('cais/mmlu','all')
        #rename mmlu features to mmlu_pro features
        ds = ds.rename_columns({"choices":"options",
                                "subject":"category"})
        
        
        datasets_to_process.append(process_ds_args | {
            "ds":ds,
            "output_dir" : OUT_DIR/ 'mmlu'
        })
    
    if get_mmlu_pro:
        ds = load_dataset('TIGER-Lab/MMLU-Pro')
        datasets_to_process.append(process_ds_args | {
            "ds":ds,
            "output_dir" : OUT_DIR/ 'mmlu_pro'
        })
        
        
    
    [process_dataset(ds) for ds in datasets_to_process]
    


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    
    allowed_langs = [ 'asm_Beng','ben_Beng','brx_Deva','doi_Deva','eng_Latn','gom_Deva','guj_Gujr',
  'hin_Deva','kan_Knda','kas_Arab','kas_Deva','mai_Deva','mal_Mlym','mar_Deva',
  'mni_Beng','mni_Mtei','npi_Deva','ory_Orya','pan_Guru','san_Deva','sat_Olck',
  'snd_Arab','snd_Deva','tam_Taml','tel_Telu','urd_Arab']
    
    
    
    parser.add_argument('-d','--dataset', 
                        choices=['mmlu', 'mmlupro', 'all'], 
                        default='all',
                        required=True,
                        help='Dataset option: "mmlu", "mmlupro", or "all"')
    parser.add_argument('-o','--output',
                        type=str,
                        default='./data',
                        help="output folder for the dataset"
                        )
    parser.add_argument('-t','--target-langs',
                        choices=allowed_langs, 
                        nargs='+',
                        default=['hin_Deva'],
                        required=True,
                        help="target languages for conversions"
                        )
    parser.add_argument('-s','--size',
                        choices=['small','large'], 
                        default='large',
                        required=False,
                        help='use the small (distilled 200M) or large (1B) model')
    parser.add_argument('-q','--quantization',
                        choices=['4','8',None], 
                        default=None,
                        required=False,
                        help='Quantization to use')
    args = parser.parse_args()
    main(args)



