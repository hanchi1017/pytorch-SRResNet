import torch.utils.data as data
import torch
import h5py
from PIL import Image
from os import listdir
from os.path import join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from io import BytesIO

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("data")
        self.target = hf.get("label")

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
        
    def __len__(self):
        return self.data.shape[0]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', 'jpeg'])

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y,_,_ = img.split()
    return y

def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])

def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])

def compress(im, quality=20):
    # https://stackoverflow.com/questions/31409506/python-convert-from-png-to-jpg-without-saving-file-to-disk-using-pil
    f = BytesIO()
    im.save(f, 'JPEG', quality=quality, optimize=True, progressive=True)
    return Image.open(f)

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, quality=20, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir,x) for x in listdir(image_dir) if is_image_file(x)]
        self.quality = quality

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()

        input = compress(input, self.quality)

        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def get_training_set(train_dir, upscale_factor):
    crop_size = calculate_valid_crop_size(128, upscale_factor)
    return DatasetFromFolder(train_dir,
                             input_transform = input_transform(crop_size, upscale_factor),
                             target_transform = target_transform(crop_size))

def get_test_set(test_dir, upscale_factor):
    crop_size = calculate_valid_crop_size(128, upscale_factor)
    return DatasetFromFolder(test_dir,
                             input_transform = input_transform(crop_size, upscale_factor),
                             target_transform = target_transform(crop_size))

if __name__ == "__main__":
    image_dir = '/home/hc/PycharmProjects/pytorch-SRResNet/data/img_align_celeba'
    dataset = DatasetFromFolder(image_dir, 20, input_transform(128,2), target_transform(128))
    input, target = dataset.__getitem__(0)
    print input