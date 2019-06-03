import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_nested_dataset
from PIL import Image
import random
import csv

CONTENTS = ['building', 'scenery', 'still-life', 'person-portrait', 'animal']

def read_content_csv(file_path):
    labels = {}
    with open(file_path) as file:
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            labels[row[0]] = int(row[1])-1  # category by filename
    return labels


class ContentLabelledDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # create a path '/path/to/data/trainA'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        # create a path '/path/to/data/trainB'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        A_images, _ = make_nested_dataset(self.dir_A, opt.max_dataset_size)    # load images from '/path/to/data/trainA'
        self.A_paths, A_contents = zip(*A_images)
        self.A_contents = [CONTENTS.index(content) for content in A_contents]
        B_images, B_labels = make_nested_dataset(self.dir_B, opt.max_dataset_size)    # load images from '/path/to/data/trainB'
        styles = sorted(B_labels.keys())
        B_labels = {style: read_content_csv(B_labels[style]) for style in styles}
        self.B_images = {i: [] for i in range(len(CONTENTS))}
        for path, style in B_images:
            content = B_labels[style][path.split('/')[-1].split('.')[0]]
            self.B_images[content].append((path, styles.index(style)))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(B_images)  # get the size of dataset A
        btoA = self.opt.direction == 'BtoA'
        # get the number of channels of input image
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        # get the number of channels of output image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        content = self.A_contents[index % self.A_size]
        # randomize the index for domain B to avoid fixed pairs.
        B_path, B_style = random.choice(self.B_images[content])
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'B_style_labels': B_style, 'content_labels': content}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
