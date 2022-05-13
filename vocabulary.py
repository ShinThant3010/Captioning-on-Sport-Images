import os
import pandas as pd
import numpy as np
import itertools
import pickle

class Vocabulary:
    def __init__(self, image_path, caption_file):
        
        # image folder path - to get image name lists
        train_images_list = os.listdir(image_path)
        
        # csv file with captions
        train_captions = pd.read_csv(caption_file, delimiter='|')
        
        # get captions
        train_captions.columns = ['image_name', 'comment_number', 'comment']
        captions = np.array(self.images_map_caption(train_images_list, train_captions))
        
        cap_len = [len(i.split()) for i in captions]
        max_cap_len = max(cap_len)
        
        ### <null> - 0, <s> - 1, <e> - 2
        start_tag = '<s>'
        end_tag = '<e>'
        null_tag = '<null>'
        
        # adjust captions length
        sentence = []
        for cap_i in captions:
            split_cap = cap_i.split()
            tagged_cap = [start_tag]
            tagged_cap.extend(split_cap)
            tagged_cap.append(end_tag)
            #print(tagged_cap)
            for _ in range(len(tagged_cap), max_cap_len+2):
                tagged_cap.append(null_tag)
            #print("length:", len(tagged_cap))
            sentence.append(tagged_cap)
        #print(sentence)

        all_vocab_list = list(itertools.chain(*sentence))
        #print(all_vocab_list)

        unique_vocab_list = list(set(all_vocab_list))
        #print(unique_vocab_list)

        null_idx = unique_vocab_list.index('<null>')
        s_idx = unique_vocab_list.index('<s>')
        e_idx = unique_vocab_list.index('<e>')

        unique_vocab_list[0], unique_vocab_list[null_idx] = unique_vocab_list[null_idx], unique_vocab_list[0]
        unique_vocab_list[1], unique_vocab_list[s_idx] = unique_vocab_list[s_idx], unique_vocab_list[1]
        unique_vocab_list[2], unique_vocab_list[e_idx] = unique_vocab_list[e_idx], unique_vocab_list[2]
        #print(unique_vocab_list)

        self.vocab_size = len(unique_vocab_list)

        # fwd_dist -> word to index
        # rev_dist -> index to word
        self.fwd_dict = {}
        self.rev_dict = {}
        for index in range(len(unique_vocab_list)):
            self.fwd_dict[unique_vocab_list[index]] = index
            self.rev_dict[index] = unique_vocab_list[index]
            
        index_cap = [[self.fwd_dict[key_word] for key_word in sentence_i] for sentence_i in sentence]
        
        # dictionary of lists 
        dict = {'img_name': train_images_list, 'Caption': index_cap} 
        df = pd.DataFrame(dict) 
        # saving the dataframe 
        df.to_csv('image_index.csv') 
        
        self.caption_lengths = [len(sen_i) for sen_i in sentence]
        
        self.vocab_file = './vocab.pkl'
        with open(self.vocab_file, 'wb') as f:
            pickle.dump(self, f)
        
    def images_map_caption(self, train_images_list, train_captions):
        caption = []
        for i in train_images_list:
            caption.append(train_captions[train_captions['image_name'] == i]['comment'].iat[0])
        return caption

