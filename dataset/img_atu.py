import torchvision.transforms as transforms

# 定义数据增强
img_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, interpolation=transforms.InterpolationMode.BICUBIC),  # Bicubic 插值
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # 颜色抖动
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),  # Random Erasing
    transforms.ToTensor(),
])

