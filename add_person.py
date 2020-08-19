import argparse
import sqlite3 as sql
from src.model import *


def add_person(fol_path, is_insider: int):
    nn = NN()
    path = fol_path.split('/')
    name = path[-1]
    if name == '':
        name = path[-2]

    con = sql.connect('dataset.db')
    query = """
    INSERT INTO employers (name, is_insider)
    VALUES (?, ?);
    """
    con.execute(query, [name, is_insider])
    query = """
    INSERT INTO embeddings (name, embedding)
    VALUES (?, ?);
    """
    image_names = os.listdir(fol_path)
    for imn in image_names:
        im_path = os.path.join(fol_path, imn)
        embedding = nn.get_embedding(im_path)
        if embedding is -1:
            continue
        con.execute(query, [name, embedding])
    con.commit()
    con.close()
    nn.close()


def main():
    parser = argparse.ArgumentParser(description='Adds all embeddings of a person to embeddings_matrix'
                                                 'Folders name will be name of a person')
    parser.add_argument('--fol_path', type=str, help='Full path to folder with images')
    parser.add_argument('--is_insider', type=int, help='Should a person be passed? 1 - Yes 0 - No')
    args = parser.parse_args()
    is_insider = args.is_insider
    fol_path = args.fol_path
    if fol_path == '':
        print('Specify the full path to folder with images')
        return
    if is_insider is None:
        print('Specify an argument is_insider')
        return
    print('It can take some time...')
    add_person(fol_path, is_insider)
    print('Successful')
    return


if __name__ == '__main__':
    main()
