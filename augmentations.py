import torch

class RandomCutout(object):
    def __init__(self, min_size_ratio=0.02,
                 max_size_ratio=0.15,
                 max_crop=10,
                 replacement=0):
        self.min_size_ratio = torch.tensor([min_size_ratio] * 2)
        self.max_size_ratio = torch.tensor([max_size_ratio] * 2)
        self.max_crop = max_crop
        self.replacement = replacement

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            size = torch.tensor(img.shape[1:3])  # H x W
        else:
            raise ValueError("Input image must be a torch tensor")

        mini = (self.min_size_ratio * size).int()
        maxi = (self.max_size_ratio * size).int()

        for _ in range(self.max_crop):
            h = torch.randint(mini[0], maxi[0], (1,)).item()
            w = torch.randint(mini[1], maxi[1], (1,)).item()

            shift_h = torch.randint(0, size[0] - h, (1,)).item()
            shift_w = torch.randint(0, size[1] - w, (1,)).item()

            img[:, shift_h:shift_h+h, shift_w:shift_w+w] = self.replacement

        return img