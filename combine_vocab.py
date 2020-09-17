from collections import Counter, OrderedDict
import json
import operator

# Combine dictionaries
count = Counter()
for i in range(1, 16):
    with open('data/all_vocab_{}.json'.format(i)) as json_file:
        data = json.load(json_file)
        count += Counter(data)

with open('data/discharge_summary_vocab.json') as json_file:
    data = json.load(json_file)
    count += Counter(data)

dic = dict(count)
sorted_dic= OrderedDict(sorted(dic.items(), key = operator.itemgetter(1), reverse = True))

# filter dic: only keep words that appear >= 20 times
keys = list(sorted_dic.keys())
values = list(sorted_dic.values())

threshold = 20
filterd_values = [value for value in values if value >= threshold]

filter_dic = dict(zip(keys[:len(filterd_values)], filterd_values))

with open('data/all_vocab.json', 'w') as fp:
    json.dump(filter_dic, fp)


def output_txt(vocab, filename):
    document = ""
    for word in vocab:
        if not word:
            continue
        
        document += word + " "
        document += str(vocab[word]) + "\n"
    
    file = open("data/{}".format(filename),"w")
    file.write(document)
    file.close()

output_txt(filter_dic, 'vocab')
