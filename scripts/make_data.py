import datasets
from datasets import load_dataset
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from IndicTransTokenizer import IndicProcessor
import warnings

def get_preds(input_samples,src_lang='eng_Latn',tgt_lang='hin_Deva',
              DEVICE='cpu', tokenizer=None,model=None, ip:IndicProcessor=None):
    with torch.device(DEVICE):
        batch = ip.preprocess_batch(
                    input_samples,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                )
        # Tokenize the sentences and generate input encodings
        inputs = tokenizer(
                    batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                )

        # Generate translations using the model
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

        # Postprocess the translations, including entity replacement
        translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        return translations




def augment_data(examples):
    outputs = []
    for sentence in examples["sentence1"]:
        words = sentence.split(' ')
        K = randint(1, len(words)-1)
        masked_sentence = " ".join(words[:K]  + [mask_token] + words[K+1:])
        predictions = fillmask(masked_sentence)
        augmented_sequences = [predictions[i]["sequence"] for i in range(3)]
        outputs += [sentence] + augmented_sequences
    return {"data": outputs}

def process_dataset(ds,,src_lang='eng_Latn',target_lang='hin_Deva',DEVICE='cpu',model=None,tokenizer=None,ip=None):
    print(ds)
    
    qs = ds['test']['question'][:3]
    
    get_preds(qs,src_lang=src_lang, tgt_lang=target_lang,
              DEVICE=DEVICE,
              model=model,
              tokenizer=tokenizer
              )
    
    

def main(args):
    get_mmlu = args.dataset in {'all','mmlu'}
    get_mmlu_pro = args.dataset in {'all','mmlupro'}
    print(args)
    print(get_mmlu, get_mmlu_pro)
    
    model_name = "ai4bharat/indictrans2-en-indic-1B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

    ip = IndicProcessor(inference=True)
    
    
    datasets_to_process=[]
    if get_mmlu:
        ds = load_dataset('cais/mmlu','all')
        #rename mmlu features to mmlu_pro features
        ds = ds.rename_columns({"choices":"options",
                                "subject":"category"})
        datasets_to_process.append(ds)
    
    if get_mmlu_pro:
        ds = load_dataset('TIGER-Lab/MMLU-Pro')
        datasets_to_process.append(ds)
        
        
    
    [process_dataset(ds,model=model,DEVICE=DEVICE,tokenizer=tokenizer,ip=ip) for ds in datasets_to_process]
    


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter
    )
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
    args = parser.parse_args()
    main(args)
