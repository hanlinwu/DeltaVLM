import torch
from transformers import GPT2Tokenizer
import argparse
import time
import json
from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]
    
    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict

def pad_tokens(tokenizer, caption):
    tokens = torch.tensor(tokenizer.encode(caption))
    padding = 50 - tokens.shape[0]
    if padding > 0:
        tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        # self.captions[item] = tokens
    elif padding < 0:
        tokens = tokens[:50]
        # self.captions[item] = tokens
    mask = tokens.ge(0)  # mask is zero where we out of sequence
    tokens[~mask] = 0
    return tokens

def eval_metrics(references_all, hypotheses_all):
    # # reference_path
    # if split_name == 'val':
    #     reference_path = r'dataset/annotations/changechat_105k_val.json'
    # elif split_name == 'test':
    #     reference_path = r'dataset/annotations/changechat_105k_test.json'

    # # tokenizer
    # gpt2_type = '/root/autodl-tmp/gpt2'
    # tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
    # # end_value = tokenizer.encode('.')[0]

    # # references
    # references_all = []
    # references_no_changed = []
    # references_changed = []
    # with open(reference_path, 'r') as j:
    #     gt_raw = json.load(j)
    # # gt_raw = [x for x in gt_raw['images'] if x['filepath'] == split_name]
    # for item in gt_raw:
    #     caption = []
    #     item['sentences'] = [x.replace(" .", "").strip() for x in item['captions']]
    #     for sentence in item['sentences']:
    #         answer = torch.tensor(tokenizer.encode(sentence)).tolist()
    #         caption.append(answer)
    #     if item['changeflag'] == 0:
    #         references_no_changed.append(caption)
    #     else:
    #         references_changed.append(caption)
    #     references_all.append(caption)


    # hypotheses_all = []
    # hypotheses_no_changed = []
    # hypotheses_changed = []
    # results = [cap['caption'] for cap in results]

    # for gt, result in zip(gt_raw, results):
    #     answer = torch.tensor(tokenizer.encode(result)).tolist()

    #     if gt['changeflag'] == 0:
    #         hypotheses_no_changed.append(answer)
    #     else:
    #         hypotheses_changed.append(answer)
        
    #     hypotheses_all.append(answer)


    # assert len(references_all) == len(hypotheses_all)
    # assert len(references_no_changed) == len(hypotheses_no_changed)
    # assert len(references_changed) == len(hypotheses_changed)

    metrics_all = get_eval_score(references_all, hypotheses_all)
    # metrics_no_changed = get_eval_score(references_no_changed, hypotheses_no_changed)
    # metrics_changed = get_eval_score(references_changed, hypotheses_changed)

    
    # print("====== no changed results ======")
    # print("BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
    #         (metrics_no_changed["Bleu_1"], metrics_no_changed["Bleu_2"], metrics_no_changed["Bleu_3"], metrics_no_changed["Bleu_4"],
    #         metrics_no_changed["METEOR"], metrics_no_changed["ROUGE_L"], metrics_no_changed["CIDEr"]))
    # print("\n")
    # print("====== changed results ======")
    # print("BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
    #         (metrics_changed["Bleu_1"], metrics_changed["Bleu_2"], metrics_changed["Bleu_3"], metrics_changed["Bleu_4"],
    #         metrics_changed["METEOR"], metrics_changed["ROUGE_L"], metrics_changed["CIDEr"]))
    
    # print("\n")
    print("====== all results ======")
    print("BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
            (metrics_all["Bleu_1"], metrics_all["Bleu_2"], metrics_all["Bleu_3"], metrics_all["Bleu_4"],
            metrics_all["METEOR"], metrics_all["ROUGE_L"], metrics_all["CIDEr"]))
    # time.sleep(10)
    return metrics_all
 
 
if __name__ == '__main__':
    path_pre = r'/root/autodl-tmp/Blip2CC/pre.json'
    path_ref = r'/root/autodl-tmp/Blip2CC/ref.json'
    results = []
    with open(path_pre, 'r') as f:
        pre = json.load(f)
    with open(path_ref, 'r') as f:
        ref = json.load(f)
    # for d in data:
    #     results.append(d['caption'])

    eval_metrics(ref, pre)