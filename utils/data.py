from collections import Counter

from .files import load_json_file_by_line, save_text_file_by_line


def load_data_from_file(file_path, max_length_text=-1):
    data = []
    for line in load_json_file_by_line(file_path):
        if max_length_text > 0:
            dict_inst = {'tokens': [w.lower() for w in line['tokens'][:max_length_text]], 'labels': line['labels'][:max_length_text]}
        else:
            dict_inst = {'tokens': [w.lower() for w in line['tokens']], 'labels': line['labels']}
        assert len(dict_inst['tokens']) == len(dict_inst['labels'])
        data.append(dict_inst)
    return data


def create_vocabulary_from_files(file_paths, voc_file_path):
    words = []
    for file_path in file_paths:
        for item in load_json_file_by_line(file_path):
            words.extend([w.lower() for w in item['tokens']])
    result = Counter(words)
    vocab = [item[0] for item in result.most_common()]
    save_text_file_by_line(voc_file_path, vocab)


if __name__ == '__main__':
    pass
