def split_data(split_ratio, data, datasets="datasets"):
    import os
    def make_file(path):
        if os.path.exists(path):
            rmtree(path)  # the directory of path is rebuild when it exist
        os.makedirs(path)
    from shutil import copy, rmtree
    import random
    random.seed(0)
    cwd = os.getcwd()
    data_root = os.path.join(cwd, datasets)   
    origin_path = os.path.join(data_root, data)
    assert os.path.exists(origin_path), 'file path: {} is not exist'.format(origin_path)
    classes = [cla for cla in os.listdir(origin_path) if os.path.join(origin_path, cla)]  # classes
    # train_dir
    train_root = os.path.join(data_root, origin_path+ '/train')
    for cla in classes:
        make_file(os.path.join(train_root, cla))
    # val_dir
    val_root = os.path.join(data_root, origin_path + '/val')
    make_file(val_root)
    for cla in classes:
        make_file(os.path.join(val_root, cla))
    for cla in classes:
        cla_path = os.path.join(origin_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        val_index = random.sample(images, k=int(num*split_ratio))
        for index, image in enumerate(images):
            if image in val_index:
                    # contribute eval samples
                    image_path = os.path.join(cla_path, image)
                    new_path = os.path.join(val_root, cla)
                    copy(image_path, new_path)
            else:
                # contribute train samples
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print('\r[{}] processing [{}/{}]'.format(cla, index+1, num), end="")
    print("\nSplit dataset Finish!")