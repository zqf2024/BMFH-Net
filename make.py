import os


def getTxt(train_dir=None, train_name=None, test_dir=None, test_name=None):
    if train_dir is not None:
        train_hazy, train_gt = train_name.split(',')
        train_hazy = os.listdir(os.path.join(train_dir, '{}/'.format(train_hazy)))
        print('创建train.txt成功')
        train_txt = open(os.path.join(train_dir, 'train.txt'), 'w+')
        for i in train_hazy:
            tmp = train_txt.writelines(i + '\n')
        train_txt.close()

    test_hazy, test_gt = test_name.split(',')
    # train_gt = os.listdir(os.path.join(train_dir, '{}/'.format(train_gt)))
    test_hazy = os.listdir(os.path.join(test_dir, '{}/'.format(test_hazy)))
    #   test_gt = os.listdir(os.path.join(test_dir, '{}/'.format(test_gt)))

    print('创建test.txt成功')
    test_txt = open(os.path.join(test_dir, 'test.txt'), 'w+')
    for i in test_hazy:
        tmp = test_txt.writelines(i + '\n')
    test_txt.close()
