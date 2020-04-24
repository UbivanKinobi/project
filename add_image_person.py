import argparse
from src.model import *


def add_image_person(im_path, name):
    nn = NN()
    if name not in nn.classes.values():
        print('No existing person name: ' + name)
        return
    key = -1
    for k, n in nn.classes.items():
        if n == name:
            key = k
            break

    embeddings = nn.get_embedding(im_path)
    if embeddings is -1:
        return
    embeddings_plus_class = np.append(embeddings, key)
    nn.insiders = np.vstack([nn.insiders, np.expand_dims(embeddings_plus_class, 0)])
    nn.save_data()
    nn.close()


def main():
    parser = argparse.ArgumentParser(description='Adds an embeddings of an image of a known person '
                                                 'to embeddings_matrix')
    parser.add_argument('--im_path', type=str, help='Full path to image')
    parser.add_argument('--name', type=str, help='Name of already saved person')
    args = parser.parse_args()
    im_path = args.im_path
    name = args.name
    if im_path == '':
        print('Specify the path to the image')
        return
    if name == '':
        print('Specify the name of a known person')
        return
    add_image_person(im_path, name)
    return


if __name__ == '__main__':
    main()
