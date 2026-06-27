from pathlib import Path
import random
import shutil


def split_class_files(
	src_class_dir: Path,
	train_class_dir: Path,
	val_class_dir: Path,
	test_class_dir: Path,
	train_ratio: float,
	val_ratio: float,
	seed: int,
):
	files = [p for p in src_class_dir.iterdir() if p.is_file()]
	files.sort()
	random.Random(seed).shuffle(files)

	total = len(files)
	train_count = int(total * train_ratio)
	val_count = int(total * val_ratio)
	test_count = total - train_count - val_count

	train_files = files[:train_count]
	val_files = files[train_count : train_count + val_count]
	test_files = files[train_count + val_count :]

	for f in train_files:
		shutil.copy2(f, train_class_dir / f.name)
	for f in val_files:
		shutil.copy2(f, val_class_dir / f.name)
	for f in test_files:
		shutil.copy2(f, test_class_dir / f.name)

	print(f"{src_class_dir.name}: total={total}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")


if __name__ == "__main__":
	data_dir = Path("./data")
	raw_dir = data_dir / "raw"

	if not raw_dir.exists():
		raise FileNotFoundError("data/raw not found")

	seed = 42
	train_ratio = 0.7
	val_ratio = 0.15

	class_names = [p.name for p in raw_dir.iterdir() if p.is_dir()]
	class_names.sort()

	for split_name in ["train", "val", "test"]:
		for class_name in class_names:
			class_dir = data_dir / split_name / class_name
			class_dir.mkdir(parents=True, exist_ok=True)
			for f in class_dir.iterdir():
				if f.is_file() and f.name != ".gitkeep":
					f.unlink()

	for class_name in class_names:
		split_class_files(
			src_class_dir=raw_dir / class_name,
			train_class_dir=data_dir / "train" / class_name,
			val_class_dir=data_dir / "val" / class_name,
			test_class_dir=data_dir / "test" / class_name,
			train_ratio=train_ratio,
			val_ratio=val_ratio,
			seed=seed,
		)

	print("Done: raw dataset copied into train/val/test")
