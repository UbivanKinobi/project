#!/usr/bin/python3.7
import argparse
import sqlite3 as sql


def delete_person(nickname: str):
    con = sql.connect('dataset.db')
    cursor = con.cursor()
    query = """SELECT COUNT(*) FROM employers;"""
    with con:
        cursor.execute(query)
        count = cursor.fetchone()[0]
    query_emp = """DELETE FROM employers WHERE nickname = '{}';""".format(nickname)
    query_emb = """DELETE FROM embeddings WHERE nickname = '{}';""".format(nickname)
    with con:
        con.execute(query_emp)
        con.execute(query_emb)
        cursor.execute(query)
        count_new = cursor.fetchone()[0]
    n = count - count_new
    if n == 0:
        print('No person was deleted')
        print('May be you specify unknown nickname')
    elif n == 1:
        print('The person: {}, was deleted from dataset.db'.format(nickname))
    else:
        print('Something REALLY wrong happens')

    con.commit()
    con.close()


def main():
    parser = argparse.ArgumentParser(description='Delete a person from dataset.db')
    parser.add_argument('--nickname', type=str, help='Nickname of already saved person')

    args = parser.parse_args()
    nickname = args.nickname

    # nickname = 'oleg'
    if nickname == '':
        print('Specify the name of of already saved person')
        return
    delete_person(nickname)
    return


if __name__ == '__main__':
    main()
