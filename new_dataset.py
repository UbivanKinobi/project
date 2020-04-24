import argparse
from src.model import *


def new_dataset(dataset_folder):
    nn = NN(load=False)
    folders = os.listdir(dataset_folder)
    for i, folder in enumerate(folders):
        nn.classes[i] = folder
        fol_path = os.path.join(dataset_folder, folder)
        image_names = os.listdir(fol_path)

        for name in image_names:
            im_path = os.path.join(fol_path, name)
            embeddings = nn.get_embedding(im_path)
            if embeddings is -1:
                continue
            embeddings_plus_class = np.append(embeddings, i)

            if nn.insiders is None:
                nn.insiders = np.expand_dims(embeddings_plus_class, 0)
            else:
                nn.insiders = np.vstack([nn.insiders, np.expand_dims(embeddings_plus_class, 0)])

    nn.save_data()
    nn.close()


def main():
    parser = argparse.ArgumentParser(description='Rewrite an embeddings_matrix and save new dataset')
    parser.add_argument('--ds_folder', type=str, help='Full path to dataset folder'
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
