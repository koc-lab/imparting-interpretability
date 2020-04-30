from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import argparse
import codecs
import time
import re
import numpy as np

def process_sentence(sentence, vocab_dict):
    sentence = sentence.split()
    inds = []
    for word in sentence:
        try:
            inds.append(vocab_dict[re.sub(r'\W+', '', word.lower())])
        except:
            pass
    return inds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors_file', required=True, type=str)
    parser.add_argument('--data_path', default='stanfordSentimentTreebank', type=str)
    parser.add_argument('--output_file', required=True, type=str)
    args = parser.parse_args()

    try:
      vectors
    except NameError: 
        print('Reading vectors file ...  ', end = '')
        t = time.time()
        with codecs.open(args.vectors_file, 'r',"UTF-8") as f:
            vocab_size = sum(1 for line in f)
        
        with codecs.open(args.vectors_file, 'r',"UTF-8") as f:
            line = f.readline()
            val = line.rstrip().split(' ')
            check = False
            if len(val)==2: # Check if the vectors file has vocab size and diensionality in the first line
                val = f.readline().rstrip().split(' ')
                vocab_size -= 1
                check = True
            vector_dim = len(list(map(float, val[1:])))
        
        vectors = np.zeros((vocab_size, vector_dim))
        
        words = [""]*vocab_size
        vocab_dict = dict()
        with codecs.open(args.vectors_file, 'r', "UTF-8") as f:
            if check:
                next(f)
            for idx, line in enumerate(f):      
                vals = line.rstrip().split(' ')
                
                words[idx] = vals[0]
                vocab_dict[vals[0]] = idx # indices start from 0
                vec = list(map(float, vals[1:]))
                try:
                    vectors[idx, :] = vec
                except IndexError:
                    if vals[0] == '<unk>':  # ignore the <unk> vector
                        pass
                    else:
                        raise Exception('IncompatibleInputs')
                        
        print("done in " + str(int(time.time() - t)) + " seconds")
        
    print('Reading train and test data ...  ', end = '')  
    t = time.time()
    dictionary = dict()
    with codecs.open(args.data_path + "/dictionary.txt", 'r',"UTF-8") as f:
        for line in f.read().splitlines():
            tmp = line.split("|")
            dictionary[tmp[0]] = int(tmp[1])
            
    with codecs.open(args.data_path + "/datasetSentences.txt","r","UTF-8") as f:
        sentences = []
        for sentence in f.read().splitlines()[1:]:     
            sentences.append(sentence.split("\t")[1])
             
    all_labels = []
    with open(args.data_path + "/sentiment_labels.txt") as f:
        for label in f.read().splitlines()[1:]:
            all_labels.append(float(label.split("|")[1]))
            
    split_classes = []  
    with open(args.data_path + "/datasetSplit.txt") as f:
        for line in f.read().splitlines()[1:]:
            split_classes.append(int(line.split(",")[1]))
            
    print("done in " + str(int(time.time() - t)) + " seconds")
    
    
    print('Generating train and test samples from the data for selected classes ...  ', end = '') 
    t = time.time()   
    
    train_size = sum([1 for label in split_classes if label == 3 or label == 1])
    test_size = sum([1 for label in split_classes if label == 2])
    
    train_samples = np.zeros([train_size, vector_dim])
    train_labels = []
    
    test_samples = np.zeros([test_size, vector_dim])
    test_labels = []
    
    train_no = 0
    test_no = 0
    not_in_dict_count = 0
    for sample_no, sentence in enumerate(sentences):
        try:
            score = all_labels[dictionary[sentence]] 
        except: 
            not_in_dict_count += 1
            continue
        
        if score <= 0.4 or score > 0.6: # Eliminate noutral sentences
            inds = process_sentence(sentence, vocab_dict)
            if len(inds) > 0:
                if split_classes[sample_no] == 1 or split_classes[sample_no] == 3:
                    for ind in inds:
                         train_samples[train_no,:] += vectors[ind,:]
                         
                    train_samples[train_no,:] = train_samples[train_no,:]/len(inds)
                    
                    if score <= 0.4:
                        train_labels.append(0)
                    elif score > 0.6:
                        train_labels.append(1)
                        
                    train_no += 1
                    
                elif split_classes[sample_no] == 2:
                    for ind in inds:
                         test_samples[test_no,:] += vectors[ind,:]
                         
                    test_samples[test_no,:] = test_samples[test_no,:]/len(inds)
                    
                    if score <= 0.4:
                        test_labels.append(0)
                    elif score > 0.6:
                        test_labels.append(1)
                    
                    test_no += 1
    
    train_samples = train_samples[:train_no,:]
    test_samples = test_samples[:test_no,:]
                 
    print("done in " + str(int(time.time() - t)) + " seconds")
      
    
    print('Training linear SVM with cross validation for parameter optimization ... ', end = '') 
    
    tuned_parameters = [{'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]    
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5)
    clf.fit(train_samples, train_labels)
    
    print("done in " + str(int(time.time() - t)) + " seconds")  
    
    predicted_labels = clf.predict(test_samples)
    accuracy = sum([true==predicted for true, predicted in zip(test_labels, predicted_labels)])/len(test_samples)*100
    
    print("Accuracy for sentiment classification of sentences is: "
          + str(round(accuracy,2)) + "% (" + str(int(accuracy/100*len(predicted_labels))) + "/" + str(len(predicted_labels)) + ")")

    f_out = open(args.output_file,"w")
    f_out.write("Accuracy for sentiment classification is: "
          + str(round(accuracy,2)) + "% (" + str(int(accuracy/100*len(predicted_labels))) + "/" + str(len(predicted_labels)) + ")\n")
    f_out.close()

if __name__ == "__main__":
    main()



 
