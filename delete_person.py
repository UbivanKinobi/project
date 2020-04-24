import argparse
from src.model import *


def delete_person(name):
    nn = NN()
    if name not in nn.classes.values():
        print('No existing person name: ' + name)
        return
    key = -1
    for k, n in nn.classes.items():
        if n == name:
            key = k
            break
    save_list = []
    for i in range(nn.insiders.shape[0]):
        if nn.insiders[i, -1] != key:
            save_list.append(i)
    nn.insiders = nn.insiders[save_list]
    del nn.classes[key]
    nn.save_data()
    nn.close()


def main():
    parser = argparse.ArgumentParser(description='Delete all embeddings of a person from embeddings_matrix')
    parser.add_argument('--name', type=str, help='Name of already saved person')
    args = parser.parse_args()
    name = args.name
    if name == '':
        print('Specify the name of of already saved person')
        return
    delete_person(name)
    print('Successful')
    return


if __name__ == '__main__':
    main()
