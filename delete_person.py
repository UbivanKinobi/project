import argparse
import sqlite3 as sql


def delete_person(name):
    con = sql.connect('dataset.db')
    query = """DELETE FROM employers WHERE name = '{}';""".format(name)
    con.execute(query)
    query = """DELETE FROM embeddings WHERE name = '{}';""".format(name)
    con.execute(query)
    con.commit()
    con.close()


def main():
    parser = argparse.ArgumentParser(description='Delete all embeddings of a person from embeddings_matrix')
    parser.add_argument('--name', type=str, help='Name of already saved person')
    args = parser.parse_args()
    name = args.name
    if name == '':
        print('Specify the name of of already saved person')
        return
    delete_person(name)
    print('Person was deleted if exists in db')
    return


if __name__ == '__main__':
    main()
