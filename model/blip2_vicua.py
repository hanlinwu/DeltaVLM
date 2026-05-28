"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from torch.nn import functional as F

import transformers
from model.blip2 import Blip2Base, disabled_train
from model.base_model import all_gather_with_grad, concat_all_gather
from model.Qformer import BertConfig, BertLMHeadModel
from utils import is_url, download_cached_file
import os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np


@torch.no_grad()
def build_connected_component(image_base, k=1):
    image_sim = torch.matmul(image_base, image_base.transpose(0, 1))
    image_norm = (image_base * image_base).sum(-1).sqrt().unsqueeze(-1)
    image_norm = torch.matmul(image_norm, image_norm.transpose(0, 1))
    dist = image_sim / image_norm  # here dist means normalized similarity

    device = dist.device
    b = dist.size(0)
    dist = dist - torch.eye(b, b, device=device) * 2
    x = torch.arange(b, device=device).unsqueeze(1).repeat(1, 1).flatten()
    y = torch.topk(dist, k, dim=1, sorted=False)[1]
    rx, ry = [], []
    for i in range(k):
        rxi = torch.cat([x, y[:, i]])
        ryi = torch.cat([y[:, i], x])
        rx.append(rxi)
        ry.append(ryi)
    rx = torch.cat(rx, 0).cpu().numpy()
    ry = torch.cat(ry, 0).cpu().numpy()
    v = np.ones(rx.shape[0])
    graph = csr_matrix((v, (rx, ry)), shape=(b, b))
    n_coms, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    labels = torch.tensor(labels, device=device)
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T)
    return mask.float()


class Blip2VicunaInstruct(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,   #改为True
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"        
        from transformers import LlamaTokenizer
        from model.modeling_llama import LlamaForCausalLM
        
        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        self.visual_encoder = self.visual_encoder.float()
        self.ln_vision = self.ln_vision.float()
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                if "blocks.37" in name or "blocks.38" in name:
                    continue
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None # 注释
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        # state_dict = self.Qformer.state_dict()
        # for name, param in self.Qformer.named_parameters():
        #     if "_query" in name:
        #         key_orig = name.replace("_query", "")
        #         param.data.copy_(state_dict[key_orig])

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(r'../vicuna-7b-v1.5', use_fast=False, truncation_side="left")
        self.llm_model = LlamaForCausalLM.from_pretrained(
            r'../vicuna-7b-v1.5', torch_dtype=torch.float16
        )
        # 0:unk 1:bos 2:eos pad_token is none; length of tokenizer is 32000
        # self.llm_tokenizer.add_special_tokens({'pad_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token
        # self.llm_model.config.pad_token_id = 0
        # self.llm_model.config.bos_token_id = 1
        # self.llm_model.config.eos_token_id = 2
        
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input

        self.context1 = nn.Linear(self.visual_encoder.num_features, self.visual_encoder.num_features, bias=False)
        self.context2 = nn.Linear(self.visual_encoder.num_features, self.visual_encoder.num_features)

        self.gate1 = nn.Linear(self.visual_encoder.num_features, self.visual_encoder.num_features, bias=False)
        self.gate2 = nn.Linear(self.visual_encoder.num_features, self.visual_encoder.num_features)

        self.dropout = nn.Dropout(0.5)

        self.context3 = nn.Linear(3*self.visual_encoder.num_features, self.visual_encoder.num_features)
        ##对比学习
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, 256)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, 256)
        self.temp = nn.Parameter(0.07 * torch.ones([]))


    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def forward(self, samples):
        # print('&&##'*50)
        # print('text_input:',samples["text_input"])
        # print('prompt:', self.prompt)
        # print('&&##'*50)
        imageA = samples["image_A"] # [b, 3, H, W]
        imageB = samples["image_B"]
        text = samples["text_input"]

        with self.maybe_autocast():
            input_bef = self.ln_vision(self.visual_encoder(imageA))
            input_aft = self.ln_vision(self.visual_encoder(imageB))
        # print(input_aft.shape)

        # image = samples["image"]
        # with self.maybe_autocast():
        #     image_embeds = self.ln_vision(self.visual_encoder(image))
        ## 差异感知部分
        # input_diff = input_aft - input_bef

        # input_bef_context = torch.tanh(self.context1(input_diff) + self.context2(input_bef))
        # input_bef_context = self.dropout(input_bef_context)
        # input_bef_gate = torch.sigmoid(self.gate1(input_diff) + self.gate2(input_bef))
        # input_bef_gate = self.dropout(input_bef_gate)
        # input_befs = input_bef_gate * input_bef_context

        # input_aft_context = torch.tanh(self.context1(input_diff) + self.context2(input_aft))
        # input_aft_context = self.dropout(input_aft_context)
        # input_aft_gate = torch.sigmoid(self.gate1(input_diff) + self.gate2(input_aft))
        # input_aft_gate = self.dropout(input_aft_gate)
        # input_afts = input_aft_gate * input_aft_context

        # input_bef = input_bef.permute(0, 2, 1)
        # input_aft = input_aft.permute(0, 2, 1)
        # # print('3'*50)
        # # print('input_befs', input_befs.shape)
        # # print('input_afts', input_afts.shape)

        # input_befs = input_befs.permute(0,2,1)
        # input_afts = input_afts.permute(0,2,1)
        # input_diff = input_diff.permute(0,2,1)

        # input_before = torch.cat([input_bef, input_diff, input_befs], 1)
        # input_after = torch.cat([input_aft, input_diff, input_afts], 1)

        # input_before = input_before.permute(0, 2, 1)
        # input_after = input_after.permute(0, 2, 1)
        # image_embedsA = self.context3(input_before)
        # image_embedsB = self.context3(input_after)
        # # print('image_embedsA', image_embedsA.shape)
        # # print('image_embedsB', image_embedsB.shape)
        # # image_embedsA = input_befs
        # # image_embedsB = input_afts
        # image_embeds = torch.cat((image_embedsA, image_embedsB), dim=1)   # [32, 514, 1408]
        ####################################################
        image_embeds = torch.cat((input_bef, input_aft), dim=1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            imageA.device
        )

        bs = imageA.size(0)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            # print('text_input','3'*50)
            # print(samples['text_input'])
            # print('text_output','3'*50)
            # print(samples['text_output'])
            text_Qformer = self.tokenizer(
                samples["text_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(imageA.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(imageA.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        ####对比学习
        # image_feats = F.normalize(
        #     self.vision_proj(query_output.last_hidden_state), dim=-1
        # )  ##[64, 32, 256]

        # text_tokens = self.tokenizer(
        #     text,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        #     return_tensors="pt",
        # ).to(imageA.device)
        
        # text_output = self.Qformer.bert(
        #     text_tokens.input_ids,
        #     attention_mask=text_tokens.attention_mask,
        #     return_dict=True,
        # )
        # text_feat = F.normalize(
        #     self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        # )

        ###============== Image-text Contrastive ===================###
        # image_feats_all = concat_all_gather(
        #     image_feats
        # )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        # text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        # sim_q2t = torch.matmul(
        #     image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        # ).squeeze()
        # # [batch_size, batch_size*num_gpu, num_query_tokens]

        # # image-text similarity: aggregate across all query tokens
        # sim_i2t, _ = sim_q2t.max(-1)
        # sim_i2t = sim_i2t / self.temp
        # # print(sim_i2t.shape)

        # # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        # sim_t2q = torch.matmul(
        #     text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        # ).squeeze()

        # # text-image similarity: aggregate across all query tokens
        # sim_t2i, _ = sim_t2q.max(-1)
        # sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]
        # # print(sim_t2i.shape)

        # ### 增加
        # # torch.autograd.set_detect_anomaly(True)
        # rank = 0
        # bs = imageA.size(0)
        # neg_mask = build_connected_component(text_feat_all)
        # # sim_t2i 
        # sim_t2i_exp = torch.exp(sim_t2i)
        # sim_t2i_sum = sim_t2i_exp.sum(1, keepdim=True)
        # normalized_sim_t2i = sim_t2i_exp / sim_t2i_sum
        # loss_infonce_t2i = normalized_sim_t2i * neg_mask
        # loss_infonce_t2i_sum = loss_infonce_t2i.sum(1)
        # loss_infonce_t2i = -torch.log(loss_infonce_t2i_sum)
        # # print(loss_infonce_t2i)
        # # sim_i2t 
        # sim_i2t_exp = torch.exp(sim_i2t)
        # sim_i2t_sum = sim_i2t_exp.sum(1, keepdim=True)
        # normalized_sim_i2t = sim_i2t_exp / sim_i2t_sum
        # loss_infonce_i2t = normalized_sim_i2t * neg_mask
        # loss_infonce_i2t_sum = loss_infonce_i2t.sum(1)
        # loss_infonce_i2t = -torch.log(loss_infonce_i2t_sum)

        # loss_itc = (loss_infonce_i2t.mean() + loss_infonce_t2i.mean())/2
        ####################################################


        # inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        # atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(imageA.device)

        # self.llm_tokenizer.padding_side = "right"
        # self.llm_tokenizer.truncation_side = 'left'

        # text_input_tokens = self.llm_tokenizer(
        #     [self.prompt]*imageA.shape[0],
        #     return_tensors="pt",
        #     padding="longest",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        # ).to(imageA.device)

        # self.llm_tokenizer.truncation_side = 'right'
        # text_output_tokens = self.llm_tokenizer(
        #     [t + self.llm_tokenizer.eos_token for t in samples['text_input']],
        #     return_tensors="pt",
        #     padding="longest",
        #     truncation=True,
        #     max_length=self.max_output_txt_len,
        # ).to(imageA.device)

        # llm_tokens, input_part_targets_len = self.concat_text_input_output(
        #     text_input_tokens.input_ids,
        #     text_input_tokens.attention_mask,
        #     text_output_tokens.input_ids,
        #     text_output_tokens.attention_mask,
        # )

        # # do not apply loss to the padding
        # targets = llm_tokens['input_ids'].masked_fill(
        #     llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        # )

        # # do not apply loss to the text input (i.e., instruction)
        # for i, l in enumerate(input_part_targets_len):
        #     targets[i][:l] = -100

        # # do not apply loss to the query tokens
        # empty_targets = (
        #     torch.ones(atts_llm.size(), dtype=torch.long).to(imageA.device).fill_(-100)
        # )
        # targets = torch.cat([empty_targets, targets], dim=1)

        # inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        # inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        # attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        # with self.maybe_autocast():
        #     outputs = self.llm_model(
        #         inputs_embeds=inputs_embeds,
        #         attention_mask=attention_mask,
        #         return_dict=True,
        #         labels=targets,
        #     )

        # loss_llm = outputs.loss
        # loss = loss_llm+loss_itc

        # return {"loss": loss, "loss_itc":loss_itc, "loss_llm":loss_llm}
        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(imageA.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(imageA.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(imageA.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(imageA.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return {"loss": loss}


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt
        # print('&&##'*50)
        # print(prompt)

        imageA = samples["image_A"] # [b, 3, H, W]
        imageB = samples["image_B"] 

        bs = imageA.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]
            # 进入
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(imageA.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(imageA.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        with self.maybe_autocast():
            input_bef = self.ln_vision(self.visual_encoder(imageA))
            input_aft = self.ln_vision(self.visual_encoder(imageB))

            input_diff = input_aft - input_bef

            input_bef_context = torch.tanh(self.context1(input_diff) + self.context2(input_bef))
            input_bef_context = self.dropout(input_bef_context)
            input_bef_gate = torch.sigmoid(self.gate1(input_diff) + self.gate2(input_bef))
            input_bef_gate = self.dropout(input_bef_gate)
            input_befs = input_bef_gate * input_bef_context

            input_aft_context = torch.tanh(self.context1(input_diff) + self.context2(input_aft))
            input_aft_context = self.dropout(input_aft_context)
            input_aft_gate = torch.sigmoid(self.gate1(input_diff) + self.gate2(input_aft))
            input_aft_gate = self.dropout(input_aft_gate)
            input_afts = input_aft_gate * input_aft_context

            input_bef = input_bef.permute(0, 2, 1)
            input_aft = input_aft.permute(0, 2, 1)

            input_befs = input_befs.permute(0,2,1)
            input_afts = input_afts.permute(0,2,1)
            input_diff = input_diff.permute(0,2,1)

            input_before = torch.cat([input_bef, input_diff, input_befs], 1)
            input_after = torch.cat([input_aft, input_diff, input_afts], 1)

            input_before = input_before.permute(0, 2, 1)
            input_after = input_after.permute(0, 2, 1)
            image_embedsA = self.context3(input_before)
            image_embedsB = self.context3(input_after)
            # image_embedsA = input_befs
            # image_embedsB = input_afts
            
            image_embeds = torch.cat((image_embedsA, image_embedsB), dim=1)   # [32, 514, 1408]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                imageA.device
            )

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(imageA.device)

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(imageA.device)

        
        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=256,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=1,
        **kwargs
    ):
        if isinstance(samples["text_input"], str):
            # 不进入
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            print('0'*50)
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                    for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            # 进入
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if 'context' in samples.keys():
                    this_sample['context'] = [samples["context"][i]]

                if 'history' in samples.keys():
                    this_sample['history'] = [samples["history"][i]]

                if 'caption' in samples.keys():
                    this_sample['caption'] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if 'context' in samples.keys() and samples['context'] != '':
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if 'history' in samples.keys() and samples['history'][0] != '':
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            # truncation=True,
            # max_length=self.max_txt_len,
        ).to(image.device)

        empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)

        # self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'
        n_cands = len(candidates)
        with self.maybe_autocast(dtype=torch.bfloat16):
            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len

                this_output_tokens = self.llm_tokenizer(
                    candidates[start_i:end_i],
                    return_tensors="pt",
                    padding="longest",
                    # truncation=True,
                    # max_length=self.max_output_txt_len,
                ).to(image.device)

                this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(seg_len, dim=0)
                this_input_tokens_atts = text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)

                this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
                this_output_tokens_atts = this_output_tokens.attention_mask.repeat(bs, 1)

                this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
                    this_input_tokens_ids,
                    this_input_tokens_atts,
                    this_output_tokens_ids,
                    this_output_tokens_atts
                )

                this_llm_input_ids = this_llm_tokens['input_ids']
                this_llm_atts = this_llm_tokens['attention_mask']
                # this_llm_input_ids = torch.cat([this_input_tokens_ids, this_output_tokens_ids], dim=1)
                # this_llm_atts = torch.cat([this_input_tokens_atts, this_output_tokens_atts], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(this_llm_input_ids)
                inputs_embeds = torch.cat([inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1)

                this_targets = this_llm_input_ids.masked_fill(this_llm_input_ids == self.llm_tokenizer.pad_token_id, -100)
                # this_targets[:, :this_input_tokens_ids.size(1)] = -100
                for i, l in enumerate(this_input_targets_len):
                    this_targets[i][:l] = -100

                this_targets = torch.cat([empty_targets.repeat_interleave(seg_len, dim=0), this_targets], dim=1)

                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )

                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)

        return output_class_ranks

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
        )

        # if qformer_text_input:
        #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
        #     model.load_from_pretrained(
        #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
        #     )

        model.load_checkpoint_from_config(cfg)

        return model
    
    # @classmethod
    # def init_Qformer(cls, num_query_token, modality_width, cross_attention_freq=2, pretrained_qformer=None, load_attention=True, load_qformer_type="image"):
    #     encoder_config = BertConfig.from_pretrained("../bert-base-cased")
    #     encoder_config.encoder_width = modality_width
    #     # insert cross-attention layer every other block
    #     encoder_config.add_cross_attention = True
    #     encoder_config.cross_attention_freq = cross_attention_freq
    #     encoder_config.query_length = num_query_token
    #     encoder_config.vocab_size += 1 # for special token [DEC]
    #     Qformer = BertLMHeadModel(config=encoder_config)
    #     query_tokens = nn.Parameter(
    #         torch.zeros(1, num_query_token, encoder_config.hidden_size)
    #     )
    #     query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

    #     pretrained_qformer="https://storage.googleapis.com/sfr-xinstructblip-data-research/model/xinstructblip_checkpoints/vicuna7b/image_qformer.pth"

    #     if pretrained_qformer:
    #         url_or_filename=pretrained_qformer
    #         logging.info(f"Loading pretrained qformer weights and query tokens from {url_or_filename} of type {load_qformer_type}")
    #         if is_url(url_or_filename):
    #             cached_file = download_cached_file(
    #                 url_or_filename, check_hash=False, progress=True
    #             )
    #             checkpoint = torch.load(cached_file, map_location="cpu")
    #         elif os.path.isfile(url_or_filename):
    #             checkpoint = torch.load(url_or_filename, map_location="cpu")
    #         else:
    #             raise RuntimeError("checkpoint url or path is invalid")
            
    #         if load_qformer_type:
    #             load_qformer_type = f"{load_qformer_type}_"
    #         loaded_state_dict = {}
    #         if 'model' in checkpoint:
    #             checkpoint = checkpoint['model'] 
    #         for k in checkpoint.keys():
    #             if load_qformer_type+'Qformer.' in k:
    #                 if not load_attention and 'attention' in k:
    #                     continue
    #                 loaded_state_dict['.'.join(k.split('.')[1:])] = checkpoint[k]
    #         Qformer.load_state_dict(loaded_state_dict, strict=False)
    #         query_tokens.data = checkpoint[load_qformer_type+'query_tokens']
        
    #     return Qformer, query_tokens