import re
import random
import os
# def preprocess_token(token, threshold):
#     token_list = token.strip().split('\t')[1:]
#     modified_list = []
#     for token in token_list:
#         new_token = re.sub('0[xX][0-9a-fA-F]+', 'MEM', token)
#         abs_values = re.findall('(?<=\s)\d+(?=\s|$)', new_token)
#         for v in abs_values:
#             # if absolute value is greater than threshhold, replace it with IMM
#             if(int(v) > threshold):
#                 new_token = new_token.replace(v, 'IMM')
#         modified_list.append(new_token)
#     return ' '.join(modified_list)

def get_dataset(data_dir):
    data_set = {}
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            data_set[file] = preprocess_file(file_path)

    #print the dataset
    # for file, token_list in data_set.items():
    #     print(f'{file}: {token_list}')

    vocabs = collect_vocab(data_set)
    # for vocab, frequency in vocabs.items():
    #     print(vocab, frequency)

    processed_dataset = min_frequency(data_set, vocabs, 2)
    for file, tokens in processed_dataset.items():
        for token in tokens:
            if(vocabs[token] < 2):
                print(token, vocabs[token])

    w2i = {word: i for i, word in enumerate(vocabs.keys())}
    i2w = {i: word for word, i in w2i.items()}
    
    return processed_dataset, vocabs, w2i, i2w

def preprocess_file(file):
    token_list = []
    with open(file, 'r') as f:
        tokens = f.readlines()
        for token in tokens:
            if "IMark" in token:
                # ignore IMark
                # token = "IMark"
                continue
            elif "AbiHint" in token:
                # and and AbiHint
                # token = "AbiHint"
                continue 
            else:
                # replace memory address
                token = re.sub('0[xX][0-9a-fA-F]+', 'MEM', token.strip())
                
                # replace offset constants
                token = re.sub('offset=\d+', 'offset=CONST', token)
                
                # replace temperary variable
                token = re.sub('t\d+', 'TVAR', token)
            #processed_token = preprocess_token(token, 1)
            token_list.append(token)
    return token_list

def collect_vocab(data_set):
    vocabs = {}
    for file, tokens in data_set.items():
        for token in tokens:
            if token not in vocabs:
                vocabs[token] = 1
            else:
                vocabs[token] += 1
    return vocabs

def min_frequency(dataset, vocabs, min_frequency):
    new_dataset = {}
    for file, tokens in dataset.items():
        new_tokens = tokens.copy()
        for token in tokens:
            #print(token)
            if(vocabs[token] < min_frequency):
                #print('remove {} {} {}'.format(file, token, vocabs[token]))
                new_tokens.remove(token)
        #print(tokens)
        new_dataset[file] = new_tokens
    return new_dataset

def get_training_data(data_set, vocab, w2i, i2w, window_size):
    idx_pairs = []
    for file, tokens in data_set.items():
        indicies = [w2i[token] for token in tokens]
        for center_word_pos in range(len(indicies)):
            contexts = []
            for w in range(-window_size, window_size+1):
                context_word_pos = center_word_pos + w

                if context_word_pos < 0 or context_word_pos >= len(indicies) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indicies[context_word_pos]
                contexts.append(context_word_idx)
                #idx_pairs.append((indicies[center_word_pos], context_word_idx))
            idx_pairs.append([contexts, indicies[center_word_pos]])
    return idx_pairs

def get_batch(data, batch_size):
    n_batches = len(data)//batch_size
    
    data = data[:n_batches*batch_size]
    for i in range(0, len(data), batch_size):
        x, y = [], []
        batch = data[i:i+batch_size]
        #print(batch)
        for j in range(len(batch)):
            y.extend(batch[j][0])
            x.extend([batch[j][1]]*len(batch[j][0]))
        yield x, y
