import os

def create_if_not_exist_rec(prefix_folders, folder):
    """
    create_if_not_exist_rec(['t1', 't2'], 'bla')
    :param prefix_folders:
    :param folder:
    :return:
    """
    if folder == "":
        return
    else:
        f = prefix_folders[-1] if len(prefix_folders) > 0 else ""
        create_if_not_exist_rec(prefix_folders[:-1], f)

        path = '{}/{}'.format('/'.join(prefix_folders), folder)
        if len(prefix_folders)>0 and folder != "" and not os.path.isdir(path):
            os.mkdir(path)

def test():
    create_if_not_exist_rec(['t1', 't2'], 'bla')

if __name__ == '__main__':
    test()