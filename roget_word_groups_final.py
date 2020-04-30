import roget
import re
from statistics import median
from statistics import mean
from itertools import groupby
import pathlib
import argparse

def get_group_info(query, dict_par2child):
    """
    Get depth information from node codes (e.g. IV, av2Fii, 18, ...)
    """
    if query == 0:
        return {'depth': 0, 'group_type': 'Root'}
    elif query in ['A', 'B', 'C', 'D', 'E', 'F']:
        return {'depth': 1, 'group_type': 'Class'}
    elif query in ['I', 'II', 'III', 'IV']:
        return {'depth': 2, 'group_type': 'Division'}
    elif query.isdigit():
        return {'depth': 3, 'group_type': 'Section'}
    elif not re.search("\d", query):
        return {'depth': 4, 'group_type': 'Subsection-depth-1'}
    elif len(query) == re.search("\d", query).start() + 1:
        return {'depth': 5, 'group_type': 'Subsection-depth-2'}
    elif len(query) == re.search("\d", query).start() + 2:
        return {'depth': 6, 'group_type': 'Subsection-depth-3'}
    elif all([x == 'i' for x in query[(re.search("\d", query).start() + 2):]]):
        return {'depth': 7, 'group_type': 'Subsection-depth-4'}
    elif query in dict_par2child:
        return {'depth': 8, 'group_type': 'Subsection-depth-5'}
    else:
        return {'depth': 9, 'group_type': 'Category'}


def gather(node, contents, dict_par2child, dict_cat2word):
    """
    A helper function that gathers all descendants for a given node.
    Accepts an empty dict as the 'contents' argument,
    updates the dict with node:descendant_list key:value pairs, for all nodes.
    """
    if node not in dict_par2child:
        bundle = dict_cat2word[node]
        contents[node] = bundle
        return bundle, contents
    else:
        merged = []
        for child in dict_par2child[node]:
            if child in contents:
                merged.extend(contents[child])
            else:
                bundle, contents = gather(child, contents, dict_par2child, dict_cat2word)
                merged.extend(bundle)
        contents[node] = list(set(merged))  # get rid of duplicates
        return merged, contents

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', required=True, type=str)
    parser.add_argument('--dim_num', required=True, type=int)
    args = parser.parse_args()
    
    # # Read Roget's Thesaurus
    r = roget.roget.Roget()
    
    # Relations between the words and the base categories
    dict_word2cat = r.thes_dict  # {'book': ['cat0811', 'cat0086', 'cat0551', 'cat0537']}
    dict_cat2word = r.thes_cat  # {'cat0811': ['accompts!', 'account', 'account book', ...]}
    
    # Relations between the base categories and their parents:
    dict_cat2node = r.basecat_parent  # bottom-up {'cat0372': 'au2'}
    dict_node2cat = r.basecat_dict  # top-down  {'ao': ['cat0274', 'cat0275']}
    
    # Remaining relations:
    dict_node2par = r.node_childparent  # bottom-up {'34': 'IV'}
    dict_par2node = r.parent_dict  # top-down {'15': ['au', 'av']}
    
    # All relations:
    dict_child2par = r.full_childparent  # bottom-up {'cat0372': 'au2', '34': 'IV'}
    dict_par2child = {}  # top-down {'15': ['au', 'av'], 'ao': ['cat0274', 'cat0275']}
    for key, value_par2node in dict_par2node.items():
        value_node2cat = dict_node2cat.get(key, None)
        if value_node2cat is not None:
            dict_par2child[key] = value_par2node + value_node2cat
        else:
            dict_par2child[key] = value_par2node
    dict_par2child = {**dict_node2cat, **dict_par2child}  # Python 3.5
    
    # Names
    cat_names = r.num_cat  # {'cat0811': 'ACCOUNTS'}
    cat_nums = r.cat_num  # {'accounts': 'cat0811'}
    node_names = r.node_codes  # {'bm': 'MONETARY RELATIONS'}
    node_codes = r.code_nodes  # {'MONETARY RELATIONS': 'bm'}
    
    
    for key, value in dict_word2cat.items():
        if 'cat16.a' in value:
            new_value = [x if x != 'cat16.a' else 'cat016.a' for x in value]
            dict_word2cat[key] = new_value
    
    cat_names['cat016.a'] = cat_names['cat16.a']
    cat_names.pop('cat16.a')
    cat_nums['nonuniformity'] = 'cat016.a'
    node_names.pop('cat16.a')
    node_codes['NONUNIFORMITY'] = 'cat016.a'
    
    dict_cat2word['cat016.a'] = dict_cat2word['cat16.a']
    dict_cat2word.pop('cat16.a')
    dict_child2par.pop('cat16.a')
    
    dict_cat2node['cat0218'] = dict_child2par['cat0218']
    dict_node2cat['ai'].insert(-1, 'cat0218')
    
    dict_cat2node['cat927.a'] = dict_child2par['cat927.a']
    
    dict_cat2word['cat0804'].remove('embarrassed')  # duplicate
    
    
    # Gather a dictionary that contains all descendants for each node
    # ('contents' .. key: node, value: descendant_list)
    contents = {}
    for node in dict_child2par:
        _, contents = gather(node, contents, dict_par2child, dict_cat2word)
    
    # Roget's Thesaurus vocabulary
    vocab_roget = sorted([word.lower() for word in list(dict_word2cat.keys())])
    
    # Read vocabulary
    with open(args.vocab_file) as vocab_file:
        vocab = [line.split()[0] for line in vocab_file.read().splitlines()]
    
    # Find the indices (according to vocab.txt) for all words in Roget's Thesaurus vocabulary
    # (not found == index None)
    vocab_roget_word_indices = {}
    perc = 5
    print_pnts = [round(len(vocab_roget)*(perc/100 + i/int(100/perc)*(1-perc/100))) for i in range(int(100/perc+1))]
    print('Checking for out-of-vocabulary words ...')
    for ind, word in enumerate(vocab_roget):
        if ind in print_pnts:
            print('Done: {}%'.format(round(ind/len(vocab_roget)*100/perc)*perc))
        try:
            vocab_roget_word_indices[word] = vocab.index(word) + 1
        except ValueError:
            vocab_roget_word_indices[word] = None
    
    # Extend descendant_list with additional info (depth, name, etc.), and turn it into a dictionary
    #
    # Also filter out out-of-vocab words.
    #
    # The resulting 'contents_filtered' is a list over such dictionaries,
    # with each element carrying information on the associated word group.
    # ('contents_filtered' elements: {...}, {'depth': 6, 'group_code': 'av2A', ...}, {...})
    #
    contents_filtered = [node for node in (dict(
        depth=get_group_info(key, dict_par2child)['depth'],
        group_code=key,
        group_name=node_names[key],
        group_type=get_group_info(key, dict_par2child)['group_type'],
        **[{'words': list(x), 'word_ids': list(y)} for x, y in  # populate words and word_ids simultaneously
           [zipped if zipped else [(), ()] for zipped in [list(zip(
               *filter(lambda x: x[1] is not None, [  # filter out-of-vocab words
                   (word_lower, vocab_roget_word_indices[word_lower])
                   for word_lower in (word.lower() for word in value)
               ])
           ))]]][0]) for key, value in contents.items()) if node['words']]
    
    # Add some more information (length of the descendant list, etc.)
    # Also sort the descendant list with respect to freq. rank
    for node in contents_filtered:
        word_ids = node['word_ids']
        node['length'] = len(word_ids)
        node['mean_freq_rank'] = mean(word_ids)
        node['median_freq_rank'] = median(word_ids)
        # Also sort the words in ascending order in freq. rank:
        t = sorted(enumerate(word_ids), key=lambda x: x[1])
        node['word_ids'] = list(list(zip(*t))[1])
        node['words'] = [node['words'][i] for i in list(zip(*t))[0]]
    
    
    # Rearrange 'contents_filtered' back into dictionary where keys = group_codes
    # ('contents_filtered_across_codes' .. key: group_code, value: dict_with_group_info)
    contents_filtered_across_codes = dict((key, list(group)[0]) for key, group in
                                          groupby(contents_filtered, key=lambda x: x['group_code']))
    
    # Rearrange 'contents_filtered_across_codes' into dictionary where keys = depth values
    # ('contents_filtered_across_depths' .. key: depth,
    # value: slice of 'contents_filtered_across_codes' with the associated depth)
    contents_filtered_across_depths = dict((key, dict(group)) for key, group in
                                           groupby(
                                               sorted(contents_filtered_across_codes.items(), key=lambda x: x[1]['depth']),
                                               key=lambda x: x[1]['depth']))
    
    
    # # Roget's Thesaurus word group determination:
    
    stack = sorted(list(zip(*contents_filtered_across_depths[3].items()))[1],
                   key=lambda x: int(x['group_code']), reverse=True)
    groups = []
    group_length_limit = 452
    
    while stack:
        node = stack.pop()
        if node['group_code'] not in dict_par2child:  # 'node' is a Category
            if node['length'] > group_length_limit:
                # Truncate the category to group_length_limit
                node['word_ids'] = node['word_ids'][:group_length_limit]
                node['words'] = node['words'][:group_length_limit]
                #
                word_ids = node['word_ids']
                node['length'] = len(word_ids)
                node['mean_freq_rank'] = mean(word_ids)
                node['median_freq_rank'] = median(word_ids)
                groups.append(node)
            else:
                groups.append(node)
        else:  # node not a Category, we may have to split
            if node['length'] > group_length_limit:
                for child in dict_par2child[node['group_code']]:
                    if child in contents_filtered_across_codes:  # skip groups that are all out-of-vocab words
                        stack.append(contents_filtered_across_codes[child])
            else:
                groups.append(node)
    print(len(groups))
    
    
    # Print a metric on the overlap of words in groups_final
    overlaps = [[len(set(group_a['word_ids']).intersection(set(group_b['word_ids'])))
                 for group_b in groups] for group_a in groups]
    
    max_mean_overlaps = max([sum(x)/len(x) for x in overlaps])
    print(max_mean_overlaps)
    
    # Also throw away small groups (so that we have ~300 groups)
    thr = 36
    groups_final = [group for group in sorted(groups, key=lambda x: x['median_freq_rank']) if group['length'] >= thr][:args.dim_num]
    print(len(groups_final))
    
    group_names = [group['group_name'] for group in groups_final]
    group_word_ids_final = [group['word_ids'] for group in groups_final]
    
    # Make sure that the output path exists
    pathlib.Path('roget_groups_out/').mkdir(exist_ok=True)
    
    
    # Will fix this with another script
    with open('roget_groups_out/forced_words_roget', 'w') as f:
        for group in group_word_ids_final:
            print(' '.join([str(i) for i in group]), file=f)
    
    # Will keep the group names
    with open('roget_groups_out/group_names', 'w') as f:
        for name in group_names:
            print(name, file=f)
            
if __name__ == "__main__":
    main()
