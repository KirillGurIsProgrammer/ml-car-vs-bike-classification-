import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms() -> transforms.Compose:
	return transforms.Compose(
		[
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
		]
	)


def create_datasets(data_dir: str = "./data", seed: int = 42):
	full_dataset = datasets.ImageFolder(root=data_dir, transform=get_transforms())

	total_size = len(full_dataset)
	train_size = int(0.7 * total_size)
	val_size = int(0.15 * total_size)
	test_size = total_size - train_size - val_size

	generator = torch.Generator().manual_seed(seed)
	train_dataset, val_dataset, test_dataset = random_split(
		full_dataset,
		lengths=[train_size, val_size, test_size],
		generator=generator,
	)

	return train_dataset, val_dataset, test_dataset


def create_dataloaders(data_dir: str = "./data", batch_size: int = 32, seed: int = 42):
	train_dataset, val_dataset, test_dataset = create_datasets(data_dir=data_dir, seed=seed)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, val_loader, test_loader


class DummyCNN(nn.Module):
	def __init__(self, num_classes: int = 2):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(32 * 56 * 56, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = self.classifier(x)
		return x


if __name__ == "__main__":
	train_loader, val_loader, test_loader = create_dataloaders(
		data_dir="./data",
		batch_size=32,
		seed=42,
	)

	model = DummyCNN(num_classes=2)

	images, labels = next(iter(train_loader))
	outputs = model(images)

	print(f"Input batch shape: {images.shape}")
	print(f"Output batch shape: {outputs.shape}")
