def data_to_xsum_format(dataset):
    if 'train' in dataset.keys():
        dataset['train'] = dataset['train'].remove_columns(['date_publish', 'source_domain', 'split'])
        dataset['train'] = dataset['train'].rename_column('title', 'summary')
        dataset['train'] = dataset['train'].rename_column('maintext', 'document')
    if 'validation' in dataset.keys():
        dataset['validation'] = dataset['validation'].remove_columns(['date_publish', 'source_domain', 'split'])
        dataset['validation'] = dataset['validation'].rename_column('title', 'summary')
        dataset['validation'] = dataset['validation'].rename_column('maintext', 'document')
    if 'test' in dataset.keys():
        dataset['test'] = dataset['test'].remove_columns(['date_publish', 'source_domain', 'split'])
        dataset['test'] = dataset['test'].rename_column('maintext', 'document')

    return dataset