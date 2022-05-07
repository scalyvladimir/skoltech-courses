import sys
import gzip
import tempfile
from io import BytesIO
from ast import literal_eval
from  urllib import request
import pandas as pd


def amazon_data_reader(path):
    with gzip.open(path, 'rt') as gz:
        for line in gz:
            yield literal_eval(line)

def read_amazon_data(path=None, name=None):
    '''Data is taken from https://jmcauley.ucsd.edu/data/amazon/'''
    if path is None and name is None:
            raise ValueError('Either the name of the dataset to download \
                or a path to a local file must be specified.')
    if path is None:
        file_url = f'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{name}_5.json.gz'
        print(f'Downloading data from: {file_url}')
        with request.urlopen(file_url) as response:
            file = response.read()
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp.write(file)
                path = temp.name
                print(f'Temporarily saved file at: {path}')
    return pd.DataFrame.from_records(
        amazon_data_reader(path),
        columns=['reviewerID', 'asin', 'overall', 'unixReviewTime']
    )

if __name__ == '__main__':
    col_names_mapping = dict(zip(
        ['reviewerID', 'asin', 'overall', 'unixReviewTime'],
        ['userid', 'itemid', 'rating', 'timestamp']
    ))
    arg = '_'.join(f'{sys.argv[1]}'.split()).replace(',', '')
    kwargs = dict.fromkeys(['path', 'name'])
    if arg.endswith('.gz'):
        kwargs['path'] = arg
    else:
        kwargs['name'] = arg
    amazon_data = (
        read_amazon_data(**kwargs)
        .rename(columns=col_names_mapping)
    )

    new_file = f'Amazon_{kwargs["name"]}_ratings_5_core.gz'
    amazon_data.to_csv(new_file, index=False)
    print(f'Saved processed file {new_file} in the current directory')