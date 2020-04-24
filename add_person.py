import argparse
from src.model import *


def add_person(fol_path):
    nn = NN()
    path = fol_path.split('/')
    name = path[-1]
    if name == '':
        name = path[-2]
    key = max(nn.classes.keys()) + 1
    nn.classes[key] = name
    image_names = os.listdir(fol_path)
    for name in image_names:
        im_path = os.path.join(fol_path, name)
        embeddings = nn.get_embedding(im_path)
        if embeddings is -1:
            continue
        embeddings_plus_class = np.append(embeddings, key)
        nn.insiders = np.vstack([nn.insiders, np.expand_dims(embeddings_plus_class, 0)])
    nn.save_data()
    nn.close()


def main():
    parser = argparse.ArgumentParser(description='Adds all embeddings of a person to embeddings_matrix'
                                                 'Folders name will be name of a person')
    parser.add_argument('--fol_path', type=str, help='Full path to folder with images')
    args = parser.parse_args()
    fol_path = args.fol_path
    if fol_path == '':
        print('Specify the full path to folder with images')
        return
    print('It can take some time...')
    add_person(fol_path)
    print('Successful')
    return


if __name__ == '__main__':
    main()
