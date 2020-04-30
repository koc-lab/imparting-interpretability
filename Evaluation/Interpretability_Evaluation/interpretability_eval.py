import argparse
import codecs
import time
import os
import numpy as np

def read_vectors(vector_file):
    with codecs.open(vector_file, 'r',"UTF-8") as f:
        vocab_size = sum(1 for line in f)
        
    with codecs.open(vector_file, 'r',"UTF-8") as f:
        line = f.readline()
        val = line.rstrip().split(' ')
        check = False
        if len(val)==2: # Check if the vectors file has vocab size and diensionality in the first line
            val = f.readline().rstrip().split(' ')
            vocab_size -= 1
            check = True
        vector_dim = len(list(map(float, val[1:])))
    
    vectors = np.zeros((vocab_size, vector_dim))
    
    vocab = [""]*vocab_size
    vocab_dict = dict()
    with codecs.open(vector_file, 'r', "UTF-8") as f:
        if check:
            next(f)
        for idx, line in enumerate(f):      
            vals = line.rstrip().split(' ')
            
            vocab[idx] = vals[0]
            vocab_dict[vals[0]] = idx # indices start from 0
            vec = list(map(float, vals[1:]))
            if len(vec) != vector_dim:
                break
            try:
                vectors[idx, :] = vec
            except IndexError:
                if vals[0] == '<unk>':  # ignore the <unk> vector
                    pass
                else:
                    raise Exception('IncompatibleInputs')
    return vocab, vectors, vocab_dict

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors_file', required=True, type=str)
    args = parser.parse_args()
    
    # Parameters
    vocab_lim = int(5e4)
    lambda_ = 5
    min_cat_word_counts = [5,10,15,20]
    Full_Vocab = ""
    #Full_Vocab = "Full_Vocab/" 
    
    # Read Vectors
    print("Loading vectors ...")
    t = time.time()
    [vocab, vectors, vocab_dict] = read_vectors(args.vectors_file)
    dim_count = len(vectors[1])
    
    if Full_Vocab == "":
        vectors = vectors[:vocab_lim,:]
        vocab = vocab[:vocab_lim]
        
    nonzero_positive_counts = np.sum(vectors > 0, 0)
    nonzero_negative_counts = np.sum(vectors < 0, 0)
    
    print("Done in " + str(int(time.time() - t)) + " seconds")
    
    nonneg = False
    if np.min(vectors) == 0:
        nonneg = True
    
    # Set paths
    category_path = 'SEMCAT'      
    # Read Categories
    cat_files = sorted(os.listdir(category_path))
    cat_count = len(cat_files)
    cat_names = []
    cat_word_counts = []
    cat_words = []
    cat_word_ids = []
    for cat_no, f_name in enumerate(cat_files):   
        cat_names.append(f_name.split("-")[0].replace("_"," "))
        cat_word_counts.append(int(f_name.split("-")[-1][:-4]))
        
        cat_words.append([])
        cat_word_ids.append([])
        
        f = open(category_path + "/" + f_name)
        for word in f.read().splitlines():
            try:
                vocab_dict[word]
                cat_words[cat_no].append(word)
                cat_word_ids[cat_no].append(vocab_dict[word])
            except:
                continue
    
    sorted_vocab_inds = []
    sorting_inds = np.argsort(vectors, axis=0)
    
    cat_positive_interpretability_scores = np.zeros([dim_count, cat_count, len(min_cat_word_counts)])
    cat_negative_interpretability_scores = np.zeros([dim_count, cat_count, len(min_cat_word_counts)])
    
    t = time.time()
    print("Evaluating interpretability ...")
    for min_cat_word_count_no, min_cat_word_count in enumerate(min_cat_word_counts):
        for cat_no in range(cat_count):
            word_ids = cat_word_ids[cat_no]
            
            max_range = lambda_*len(word_ids)      
            for dim_no in range(dim_count):
                           
                if nonzero_positive_counts[dim_no] != 0:             
                    if nonzero_positive_counts[dim_no] < max_range:
                        true_max_range = nonzero_positive_counts[dim_no]
                    else:
                        true_max_range = max_range
                      
                    Match_positive = list(set(word_ids) & set(sorting_inds[-true_max_range:, dim_no])) 
                    positive_indices = sorted([len(vocab) - true_max_range + np.where(sorting_inds[-true_max_range:, dim_no] == positive_index)[0][0] for positive_index in Match_positive])
                    
                    if len(positive_indices) >= min_cat_word_count:
                        for i in range(len(word_ids),min_cat_word_count-1,-1):
                            edge_word_count = sum([1 for ind in positive_indices  if (ind >= (len(vocab)-lambda_*i))])
                            
                            if edge_word_count >= min_cat_word_count:
                                score = edge_word_count/i*100
                                
                                if score > cat_positive_interpretability_scores[dim_no, cat_no, min_cat_word_count_no]:
                                    cat_positive_interpretability_scores[dim_no, cat_no, min_cat_word_count_no] = min(score,100)                     
                            else:
                                break
     
                if not nonneg:                    
                    if nonzero_negative_counts[dim_no] != 0:               
                        if nonzero_negative_counts[dim_no] < max_range:
                            true_max_range = nonzero_negative_counts[dim_no]
                        else:
                            true_max_range = max_range
                          
                        Match_negative = list(set(word_ids) & set(sorting_inds[:true_max_range, dim_no])) 
                        negative_indices = sorted([np.where(sorting_inds[:true_max_range, dim_no] == negative_index)[0][0] for negative_index in Match_negative])
            
                        if len(negative_indices) >= min_cat_word_count:
                            for i in range(len(word_ids),min_cat_word_count-1,-1):
                                edge_word_count = sum([1 for ind in negative_indices  if (ind < lambda_*i)])
            
                                if edge_word_count >= min_cat_word_count:
                                    score = edge_word_count/i*100
                                    
                                    if score > cat_negative_interpretability_scores[dim_no, cat_no, min_cat_word_count_no]:
                                        cat_negative_interpretability_scores[dim_no, cat_no, min_cat_word_count_no] = min(score,100)
                                else:
                                    break
    
        print("Minimum category word count: " + str(min_cat_word_count) + ", Elapsed: " + str(int((time.time() - t)/60)) + " minutes " + str(int((time.time() - t) %60)) + " seconds" )
    
    cat_interpretability_scores_for_each_dim = np.maximum(cat_positive_interpretability_scores, cat_negative_interpretability_scores)
    interpretability_scores_for_dimensions = np.amax(cat_interpretability_scores_for_each_dim, axis=1)
    average_interpretability_scores = np.mean(interpretability_scores_for_dimensions,axis=0)
    for min_cat_word_count_no, min_cat_word_count in enumerate(min_cat_word_counts):
        f = codecs.open("interpretability_results_n_min_" + str(min_cat_word_count) + ".txt", 'a+',"UTF-8")
        f.write("interpretability: " + str(round(average_interpretability_scores[min_cat_word_count_no],2)))
        f.close()


if __name__ == "__main__":
    main()
