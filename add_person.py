#!/usr/bin/python3.7
import os
import argparse
import sqlite3 as sql
from src.model import NN


def add_person(fol_path: str, nickname: str, firstname: str, lastname: str, is_insider: int):
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
    query = """
    INSERT INTO employers (nickname, firstname, lastname, is_insider)
    VALUES (?, ?, ?, ?);
    """
    with con:
        con.execute(query, [nickname, firstname, lastname, is_insider])
    query = """
    INSERT INTO embeddings (nickname, embedding)
    VALUES (?, ?);
    """
    image_names = os.listdir(fol_path)
    for imn in image_names:
        im_path = os.path.join(fol_path, imn)
        embedding = nn.get_embedding(im_path)
        if embedding is -1:
            continue
        with con:
            con.execute(query, [nickname, embedding])

    con.commit()
    nn.close()


def main():
    parser = argparse.ArgumentParser(description='Adds a person into dataset.db')
    parser.add_argument('--fol_path', type=str, help='Full path to folder with images')
    parser.add_argument('--nickname', type=str, help='Nickname must be unique')
    parser.add_argument('--firstname', type=str, help='Firstname of your person')
    parser.add_argument('--lastname', type=str, help='Lastname of your person')
    parser.add_argument('--is_insider', type=int, help='Should a person be passed? 1 - Yes 0 - No')

    args = parser.parse_args()
    nickname = args.nickname
    firstname = args.firstname
    lastname = args.lastname
    fol_path = args.fol_path
    is_insider = args.is_insider

    # nickname = 'oleg'
    # firstname = 'oleg'
    # lastname = 'grigoriev'
    # fol_path = 'src/dataset/oleg'
    # is_insider = 1
    if fol_path == '':
        print('Specify the full path to folder with images')
        return
    elif nickname == '':
        print('Specify a nickname')
        return
    elif nickname.lower == 'alien':
        print('Alien is a reserved nickname')
        return
    elif firstname == '':
        print('Specify a firstname')
        return
    elif lastname == '':
        print('Specify a lastname')
        return
    elif is_insider is None:
        print('Specify an argument is_insider')
        return
    print('It can take some time...')
    add_person(fol_path, nickname, firstname, lastname, is_insider)
    print('Successful')
    return


if __name__ == '__main__':
    main()
