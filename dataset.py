import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvsion import transforms

class CarvanaDataset(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        if test: #want to load from the manual directories directly
            self.images = sorted([root_path + "/manual_test/" + i for i in os.listdir(root_path+"/manual_test/")])
            self.masks = sorted([root_path + "/manual_test_masks/" + i for i in os.listdir(root_path+"/manual_test_masks/")])

        self.transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])   

    def __getitem(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.images[index]).convert("L") #1 channel
        
        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)

