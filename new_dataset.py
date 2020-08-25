import os
import json
import argparse
import sqlite3 as sql
from src.model import NN


def new_dataset(dataset_folder, json_path):
    nn = NN(load=False)
    con = sql.connect('dataset.db')
    query_emp = """
    CREATE TABLE IF NOT EXISTS employers (
    nickname TINYTEXT NOT NULL PRIMARY KEY,
    firstname TINYTEXT,
    lastname TINYTEXT,
    is_insider TINYINT);
    """
    query_emb = """
    CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    nickname TINYTEXT NOT NULL,
    embedding BLOB,
    FOREIGN KEY (nickname) REFERENCES employers(nickname));
    """
    query_log = """
    CREATE TABLE IF NOT EXISTS access_control_log (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    nickname TINYTEXT NOT NULL,
    was_passed TINYINT, 
    photo BLOB,
    datetime DATETIME,
    dist_1 DOUBLE,
    dist_2 DOUBLE,
    dist_3 DOUBLE,
    dist_4 DOUBLE,
    dist_5 DOUBLE,
    FOREIGN KEY (nickname) REFERENCES employers(nickname));
    """
    with con:
        con.execute(query_emp)
        con.execute(query_emb)
        con.execute(query_log)
    query_emp = """
    INSERT INTO employers (nickname, firstname, lastname, is_insider)
    VALUES (?, ?, ?, ?);
    """
    query_emb = """
    INSERT INTO embeddings (nickname, embedding)
    VALUES (?, ?);
    """

    with open(json_path, 'r') as f:
        s = f.read()
    description = json.loads(s)
    folders = os.listdir(dataset_folder)
    for folder in folders:
        desc = description[folder]
        with con:
            data = [desc['nickname'], desc['firstname'], desc['lastname'], desc['is_insider']]
            con.execute(query_emp, data)

        fol_path = os.path.join(dataset_folder, folder)
        image_names = os.listdir(fol_path)
        for imn in image_names:
            im_path = os.path.join(fol_path, imn)
            embedding = nn.get_embedding(im_path)
            if embedding is -1:
                continue
            with con:
                data = [desc['nickname'], embedding]
                con.execute(query_emb, data)

    con.commit()
    nn.close()


def main():
    parser = argparse.ArgumentParser(description='Create dataset.db and fill it with some information about employers. '
                                                 'Also calculate feature vectors of each picture.')
    parser.add_argument('--ds_folder', type=str, help='Full path to dataset folder. '
                                                      'Names of internal folders will be used as identifiers of persons'
                                                      ' in dataset.json')
    parser.add_argument('--json_path', type=str, help='Full path to {filename}.json which should contain description of'
                                                      ' each person folder in dataset folder.')
    args = parser.parse_args()
    ds_folder = args.ds_folder
    json_path = args.json_path

    # ds_folder = 'src/dataset'
    # json_path = 'src/dataset.json'
    if ds_folder == '':
        print('Specify the full path to dataset folder')
        return
    elif json_path == '':
        print('Specify the full path to {filename}.json')
        return
    print('It can take some time...')
    new_dataset(ds_folder, json_path)
    print('Successful')
    return


if __name__ == '__main__':
    main()
