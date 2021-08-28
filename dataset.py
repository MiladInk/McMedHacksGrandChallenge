import torch
from torch.utils.data import DataLoader, Dataset

class AlphaTauDataset(Dataset):
    def __init__(self, path):
        import os
        self.path = path
        self.dirs = sorted(os.listdir(path))
        super(AlphaTauDataset, self).__init__()

    def __getitem__(self, index):
        file = self.path + self.dirs[index]
        mask = np.load(file + '/masks/mask.npy')
        data = np.load(file + '/images/' + self.dirs[index] + '.npy')
        seeds = np.load(file + '/seeds.npy')

        import math
        tdata = torch.tensor(data.astype('float32')).permute(2, 0, 1)
        tmask = torch.tensor(mask > 0).float().permute(2, 0, 1)
        width = tdata.shape[1]
        height = tdata.shape[2]
        desired_width = math.ceil(width / 16) * 16
        desired_height = math.ceil(height / 16) * 16
        pad_dims = (math.ceil((desired_width - width) / 2), math.ceil((desired_height - height) / 2),
                    math.floor((desired_width - width) / 2), math.floor((desired_height - height) / 2))
        tdata = torchvision.transforms.Pad(pad_dims, fill=-1000., padding_mode='constant')(tdata)
        tmask = torchvision.transforms.Pad(pad_dims, fill=0, padding_mode='constant')(tmask)

        # adding extra empty channels to make all inputs have 41 channels
        to_add_channels = 45 - tdata.shape[0]
        left = 2
        right = 45 - tdata.shape[0] - left
        tdata = torch.cat((torch.ones(left, tdata.shape[1], tdata.shape[2]) * -1000., tdata,
                           torch.ones(right, tdata.shape[1], tdata.shape[2]) * -1000.), axis=0)
        tmask = torch.cat((tmask, torch.zeros(41 - tmask.shape[0], tmask.shape[1], tmask.shape[2])), axis=0)

        mu = tdata.reshape(-1).mean()
        st = tdata.reshape(-1).std()
        tdata = (tdata - mu) / st

        return tdata, tmask, torch.tensor(seeds)

    def __len__(self):
        return len(self.dirs)


class TestAlphaTauDataset(Dataset):
    def __init__(self, path):
        import os
        self.path = path
        self.dirs = sorted(os.listdir(path))
        self.dirs = [x for x in self.dirs if x.find('.DS_Store') == -1]
        super(TestAlphaTauDataset, self).__init__()

    def __getitem__(self, index):
        file = self.path + self.dirs[index]
        data = np.load(file + '/images/' + self.dirs[index] + '.npy')

        import math
        tdata = torch.tensor(data.astype('float32')).permute(2, 0, 1)
        width = tdata.shape[1]
        height = tdata.shape[2]
        desired_width = math.ceil(width / 16) * 16
        desired_height = math.ceil(height / 16) * 16
        pad_dims = (math.ceil((desired_width - width) / 2), math.ceil((desired_height - height) / 2),
                    math.floor((desired_width - width) / 2), math.floor((desired_height - height) / 2))
        tdata = torchvision.transforms.Pad(pad_dims, fill=-1000., padding_mode='constant')(tdata)

        # adding extra empty channels to make all inputs have 41 channels
        to_add_channels = 45 - tdata.shape[0]
        left = 2
        right = 45 - tdata.shape[0] - left
        tdata = torch.cat((torch.ones(left, tdata.shape[1], tdata.shape[2]) * -1000., tdata[0:41],
                           torch.ones(right, tdata.shape[1], tdata.shape[2]) * -1000.), axis=0)

        mu = tdata.reshape(-1).mean()
        st = tdata.reshape(-1).std()
        tdata = (tdata - mu) / st

        return tdata

    def __len__(self):
        return len(self.dirs)