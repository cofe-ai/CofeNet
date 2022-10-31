import os
import json

PARDIR = os.path.pardir


def define_dir(*args):
    folder = os.path.abspath(os.path.join(*args))
    if not os.path.exists(folder):
        os.makedirs(folder)
    assert os.path.isdir(folder), '[ERROR] \'%s\' is not a folder path!' % folder
    return folder


def concat_path(*args):
    return os.path.abspath(os.path.join(*args))


def cur_dir_abspath(cur_file):
    return os.path.abspath(os.path.dirname(cur_file))


def find_all_file_paths(src_dir):
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(src_dir)
    for dir_cur, _, file_names in os.walk(src_dir, topdown=False):
        for file_name in file_names:
            if file_name.startswith('.'):
                continue
            yield os.path.join(dir_cur, file_name), file_name


def find_cur_file_paths(src_dir):
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(src_dir)
    for file_name in os.listdir(src_dir):
        file_path = os.path.join(src_dir, file_name)
        if os.path.isfile(file_path):
            yield file_path, file_name


def get_file_paths(src_dir, sub_dir=True):
    iter_fun = find_all_file_paths if sub_dir else find_cur_file_paths
    file_infos = [x for x in iter_fun(src_dir)]
    if file_infos:
        file_paths, file_names = zip(*[x for x in iter_fun(src_dir)])
        return list(file_paths), list(file_names)
    else:
        return [], []


def load_text_file(file_path) -> str:
    with open(file_path, 'r', encoding='UTF-8') as f:
        data = f.read()
    return data


def save_text_file(file_path, data: str):
    with open(file_path, 'w', encoding='UTF-8') as f:
        f.write(data)


def load_text_file_by_line(file_path):
    with open(file_path, 'r', encoding='UTF-8') as f:
        data = [token.replace('\n', '').replace('\r', '') for token in f.readlines()]
    return [x for x in data if x]


def save_text_file_by_line(file_path, data: list):
    with open(file_path, 'w', encoding='UTF-8') as f:
        f.write('\n'.join(data))


def load_json_file(file_path):
    with open(file_path, 'r', encoding='UTF-8') as f:
        org_data = json.load(f)
    return org_data


def save_json_file(file_path, data, indent=None):
    with open(file_path, 'w', encoding='UTF-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json_file_by_line(file_path):
    return [json.loads(line) for line in load_text_file_by_line(file_path)]


def save_json_file_by_line(file_path, data: list):
    save_text_file_by_line(file_path, [json.dumps(x, ensure_ascii=False) for x in data])


def load_data_file(file_path: str):
    usrs, prds, labels, docs = [], [], [], []
    for line in load_text_file_by_line(file_path):
        items = line.split('\t\t')
        usrs.append(items[0])
        prds.append(items[1])
        labels.append(int(items[2]) - 1)
        docs.append([sent.strip().split(' ') for sent in items[3][0:-1].split('<sssss>')])
    return usrs, prds, labels, docs


if __name__ == '__main__':
    pass
