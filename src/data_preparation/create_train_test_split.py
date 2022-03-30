import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        default='/home/bianca/PycharmProjects/MultimodalGlosses/data/in/word+img.train.dev.test.dm.txt')
    parser.add_argument("--out_file", type=str,
                        default='/home/bianca/PycharmProjects/MultimodalGlosses/data/in/word+img.{}.dm.txt')
    parser.add_argument("--perc_train", type=int,
                        default=80)

    args = parser.parse_args()
    input_file = args.input_file
    out_file = args.out_file
    perc_train = args.perc_train

    dataset = list()
    with open(input_file, 'rt') as lines:
        for line in lines:
            dataset.append(line)

    dataset_len = len(dataset)
    num_train_ex = int((len(dataset)*perc_train)/100)
    num_dev_ex = int((len(dataset)*((100-(perc_train))/2))/100)
    res = dataset_len - ((num_train_ex) + num_dev_ex*2)

    num_train_ex += res

    assert num_train_ex+num_dev_ex*2 == dataset_len

    dataset_idxs = np.arange(dataset_len)
    np.random.shuffle(dataset_idxs)

    train_ex = set(np.random.choice(dataset_idxs, num_train_ex, replace=False)) 
    
    dataset_idxs = [x for x in dataset_idxs if x not in train_ex]
    dev_ex = set(np.random.choice(dataset_idxs, num_dev_ex, replace=False))

    test_ex = [x for x in dataset_idxs if x not in dev_ex]

    datasets = {'train': train_ex, 'dev': dev_ex, 'test': test_ex}

    for i in datasets:
        with open(out_file.format(i), 'wt') as writer:
            for ex in datasets[i]:
                writer.write(dataset[ex])


    

