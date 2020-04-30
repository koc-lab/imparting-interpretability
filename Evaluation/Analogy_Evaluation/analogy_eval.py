#!/usr/bin/env python2
import argparse
import numpy as np
import inspect
import os
import codecs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors_file', required=True, type=str)
    parser.add_argument('--qa_dataset', default='Question_Data', type=str)
    parser.add_argument('--output_file', required=True, type=str)
    args = parser.parse_args()
    
    print('Reading vectors file ...')
    with open(args.vectors_file, 'r') as f:
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
    
    W = np.zeros((vocab_size, vector_dim))
   
    words = [""]*vocab_size
    with open(args.vectors_file, 'r') as f:
        if check:
            next(f)
        for idx, line in enumerate(f):      
            vals = line.rstrip().split(' ')
            words[idx] = vals[0]
            vec = list(map(float, vals[1:]))
            try:
                W[idx, :] = vec
            except IndexError:
                if vals[0] == '<unk>':  # ignore the <unk> vector
                    pass
                else:
                    raise Exception('IncompatibleInputs')
    print('Done.\n')
            

    print('Checking for duplicates ...')
    seen = set()
    seen_add = seen.add
    vocab = {w.lower(): idx for idx, w in enumerate(words) if not (w.lower() in seen or seen_add(w.lower()))}
    
    print('Done.\n')

    print('Normalizing ...')

    d = (np.sum(W ** 2, 1) ** (0.5))
    np.place(d, d == 0, 1) # If there is an all zero vector prevent dividing by zero
    W_norm = (W.T / d).T
    print('Done.\n\nEvaluating ...')
    evaluate_vectors(W_norm, vocab, words, args.qa_dataset, args.output_file)

def evaluate_vectors(W, vocab, words, prefix, output_file):
    """Evaluate the trained word vectors on a variety of tasks"""

    filenames = [file for file in os.listdir(prefix)
                 if os.path.isfile(os.path.join(prefix, file)) and file.endswith('.txt')]
    filenames.sort()

    split_size = 100

    correct_sem = 0  # count correct semantic questions
    correct_syn = 0  # count correct syntactic questions
    correct_tot = 0  # count correct questions
    count_sem = 0  # count all semantic questions
    count_syn = 0  # count all syntactic questions
    count_tot = 0  # count all questions
    full_count = 0  # count all questions, including those with unknown words

    words = [word.lower() for word in words]

    with open(output_file, 'a') as outf:
        for i in range(len(filenames)):
            with open('%s/%s' % (prefix, filenames[i]), 'r') as f:
                full_data = [line.rstrip().split(' ') for line in f]
                full_count += len(full_data)
                data = [x for x in full_data if all(word in vocab for word in x)]

            indices = np.array([[vocab[word.lower()] for word in row] for row in data])
            answer_words = [row[3].lower() for row in data]
            ind1, ind2, ind3, ind4 = indices.T

            predictions = np.zeros((len(indices),))
            num_iter = int(np.ceil(len(indices) / float(split_size)))
            for j in range(num_iter):
                subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))

                pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
                    +  W[ind3[subset], :])
                #cosine similarity if input W has been normalized
                dist = np.dot(W, pred_vec.T)

                for k in range(len(subset)):
                    dist[ind1[subset[k]], k] = -np.Inf
                    dist[ind2[subset[k]], k] = -np.Inf
                    dist[ind3[subset[k]], k] = -np.Inf

                # predicted word index
                predictions[subset] = np.argmax(dist, 0).flatten()

            prediction_ids = [int(x) for x in predictions.tolist()]
            predicted_words = [words[idx] for idx in prediction_ids]
            val = [w1 == w2 for (w1,w2) in zip(answer_words, predicted_words)]
            # val_old = (ind4 == predictions) # correct predictions

            count_tot = count_tot + len(ind1)
            correct_tot = correct_tot + sum(val)
            if not filenames[i].startswith('gram'):
                count_sem = count_sem + len(ind1)
                correct_sem = correct_sem + sum(val)
            else:
                count_syn = count_syn + len(ind1)
                correct_syn = correct_syn + sum(val)

            print("%s:" % filenames[i])
            outf.write("%s:\n" % filenames[i])
            print('ACCURACY TOP1: %.2f%% (%d/%d)' %
                (np.mean(val) * 100, np.sum(val), len(val)))
            outf.write('ACCURACY TOP1: %.2f%% (%d/%d)\n' %
                (np.mean(val) * 100, np.sum(val), len(val)))

        print('Questions seen/total: %.2f%% (%d/%d)' %
            (100 * count_tot / float(full_count), count_tot, full_count))
        outf.write('\nQuestions seen/total: %.2f%% (%d/%d)\n' %
            (100 * count_tot / float(full_count), count_tot, full_count))
        print('Semantic accuracy: %.2f%%  (%i/%i)' %
            (100 * correct_sem / float(count_sem), correct_sem, count_sem))
        outf.write('Semantic accuracy: %.2f%%  (%i/%i)\n' %
            (100 * correct_sem / float(count_sem), correct_sem, count_sem))
        print('Syntactic accuracy: %.2f%%  (%i/%i)' %
            (100 * correct_syn / float(count_syn), correct_syn, count_syn))
        outf.write('Syntactic accuracy: %.2f%%  (%i/%i)\n' %
            (100 * correct_syn / float(count_syn), correct_syn, count_syn))
        print('Total accuracy: %.2f%%  (%i/%i)' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))
        outf.write('Total accuracy: %.2f%%  (%i/%i)\n\n' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))


if __name__ == "__main__":
    main()
