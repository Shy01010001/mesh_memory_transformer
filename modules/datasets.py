import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import spacy
# def min_max_norm(data_dict):
#     minm = 1000.
#     maxm = .0
#     for key in data_dict:
#         if data_dict[key] < minm:
#             minm = data_dict[key]
#         if data_dict[key] > maxm:
#             maxm = data_dict[key]
#     con = maxm - minm
#     for key in data_dict:
#         data_dict[key] = (data_dict[key] - minm) / con
#     return data_dict

        
# def cal_tf_idf(documents, tokenizer):
#     sents_num = len(documents)
#     dic_tf = {}
#     dic_idf = {}
#     tf_idf = {}
#     for i in range(sents_num):
#         count = 0
#         sents = documents[i].split()
#         dic = {}
#         dic_tmp_tf = {}
#         for j in sents:
#             x = tokenizer.get_id_by_token(j)
#             if x not in dic_tmp_tf:
#                 dic_tmp_tf[x] = 1
#             else:
#                 dic_tmp_tf[x] += 1
        
#             dic[x] = 1
#             count += 1
#         for j in dic_tmp_tf:
#             # print(j)
#             if j not in dic_tf:
#                 # dic_tf[j] = [np.log(1 + dic_tmp_tf[j]/count)] ### log
#                 dic_tf[j] = [dic_tmp_tf[j]/count]
#             else:
#                 # dic_tf[j].append(np.log(1 + dic_tmp_tf[j]/count))
#                 dic_tf[j].append(dic_tmp_tf[j]/count)
        
             
#         for j in dic:
#             if j not in dic_idf:
#                 dic_idf[j] = 1
#             else:
#                 dic_idf[j] += 1
                
#     for i in dic_idf:
#         dic_idf[i] = np.log(sents_num / dic_idf[i])
#     # print(dic_tf)
#     for i in dic_tf:
#         s = 0
#         for j in dic_tf[i]:
#             s += j
#         dic_tf[i] = s / len(dic_tf[i])
#     # print(dic_tf)
#     for i in dic_idf:
#         tf_idf[i] = dic_idf[i] * dic_tf[i]
#     # print(dic_idf)
#     # print()
#     # print(dic_tf)
#     # print()
#     # print(tf_idf[1])
#     tf_idf = min_max_norm(tf_idf)
#     # print()
#     # print(tf_idf)
#     for i in tf_idf:
#         if tf_idf[i] < 0.1:
#             print(tokenizer.get_token_by_id(i))
#     exit()
#     # print(dic_tf)
#     # draw_plot(dic_tf)
#     # exit()
#     result = []
#     for i in range(sents_num):
#         result.append([])
#         sents = documents[i].split()
#         for j in sents:
#             result[-1].append(dic_tf[j] * dic_idf[j])

def pos_tagging(documents, tokenizer):
    nlp = spacy.load("en_core_web_sm")  # 加载英文模型
    pos_dict = {}  # 存储单词和词性的字典
    # pos_word_dict = {}  # 存储词性和对应单词列表的字典
    
    for doc in documents:
        doc = nlp(doc)  # 对文档进行处理

        for token in doc:
            word = token.text
            pos = token.pos_

            pos_dict[tokenizer.get_id_by_token(word)] = pos
            

            # if pos in pos_word_dict:
            #     if word not in pos_word_dict[pos]:
            #         pos_word_dict[pos].append(word)
            # else:
            #     pos_word_dict[pos] = [word]
    # with open('./data/iu_xray/pos_dict.json', 'w') as f:
        # json.dump(pos_dict, f)
    # print(pos_dict)
    # exit()
    # return pos_dict



class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length # 60
        self.split = split # train/valid/test
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        # if self.split =='train':
        #     self.examples = self.ann[self.split] + self.ann['test']
        # else:
        #     self.examples = self.ann['val']
        # self.tf_idf = json.loads(open('./data/iu_xray/tf_idf.json', 'r').read())[self.split]
        
        # dd = {}
        # documents = []
        
        # for key in self.ann:
            # for dic in self.ann[key]:
                # documents.append(self.tokenizer.clean_report_iu_xray(dic['report']))    
        # speach_dict = pos_tagging(documents, self.tokenizer)
        # dd['train'] = tf_idf[:2069]
        # dd['val'] = tf_idf[2069:2069+296]
        # dd['test'] = tf_idf[2069+296:2069+296+590]
        # save_path = './data/iu_xray/ids_2_tf_idf.json'
        # with open(save_path, "w") as f:
        #     json.dump(dd, f)
        # exit()
        for i in range(len(self.examples)): # 每个例子
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            # self.examples[i]['tf_idf'] = self.tf_idf[i][:self.max_seq_length]
    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        

        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1) # [3, 224, 224]
            image_2 = self.transform(image_2)

        image = torch.stack((image_1, image_2), 0) # [2图片, 3, 224, 224]
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids) 
        # sample = (image_id, image, report_ids, report_masks, seq_length, tf_idf)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_id = os.path.join(self.image_dir, image_path[0])
        if self.transform is not None:
            image = self.transform(image) # [3, 224, 224]

        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample