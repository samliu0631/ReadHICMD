import os

def prepare_sub_folder_pseudo(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    pseudo_directory = os.path.join(output_directory, 'pseudo_train')
    if not os.path.exists(pseudo_directory):
        print("Creating directory: {}".format(pseudo_directory))
        os.makedirs(pseudo_directory)
        os.makedirs(pseudo_directory + '/train_all')
    else:
        rmtree(pseudo_directory)
        os.makedirs(pseudo_directory)
        os.makedirs(pseudo_directory + '/train_all')

    return checkpoint_directory, image_directory, pseudo_directory



def extractfeatures():
    pass