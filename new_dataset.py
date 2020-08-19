import argparse
import sqlite3 as sql
from src.model import *


def new_dataset(dataset_folder):
    nn = NN(load=False)
    con = sql.connect('dataset.db')
    query = """
    CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    name TINYTEXT,
    embedding BLOB );
    """
    with con:
        con.execute(query)
    query = """
    CREATE TABLE IF NOT EXISTS employers (
    name TINYTEXT NOT NULL PRIMARY KEY,
    is_insider TINYINT );
    """
    with con:
        con.execute(query)
    query_emb = """
    INSERT INTO embeddings (name, embedding)
    VALUES (?, ?);
    """
    query_emp = """
    INSERT INTO employers (name, is_insider)
    VALUES (?, ?);
    """

    separation = ('insiders', 'external')
    for sp in separation:
        sp_path = os.path.join(dataset_folder, sp)
        folders = os.listdir(sp_path)
        for folder in folders:
            fol_path = os.path.join(sp_path, folder)
            image_names = os.listdir(fol_path)
            with con:
                is_insider = 0
                if sp == 'insiders':
                    is_insider = 1
                data = [folder, is_insider]
                con.execute(query_emp, data)

            for imn in image_names:
                im_path = os.path.join(fol_path, imn)
                embedding = nn.get_embedding(im_path)
                if embedding is -1:
                    continue
                with con:
                    data = [folder, embedding]
                    con.execute(query_emb, data)\

    con.close()
    nn.close()


def main():
    parser = argparse.ArgumentParser(description='Rewrite an embeddings_matrix and save new dataset')
    parser.add_argument('--ds_folder', type=str, help='Full path to dataset folder. '
                                                      'Names of internal folders will be used as identifiers of persons'
                        )
    args = parser.parse_args()
    ds_folder = args.ds_folder
    if ds_folder == '':
        print('Specify the full path to dataset folder')
        return
    print('It can take some time...')
    new_dataset(ds_folder)
    print('Successful')
    return


if __name__ == '__main__':
    main()
