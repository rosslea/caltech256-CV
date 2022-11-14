import math
from pathlib import Path


# Get all files in the current directory
def split_files(oldpath, newpath, classes):
    if classes is None:
      classes = [str(i.name) for i in (Path.cwd()/Path(oldpath)).iterdir()]
      print(classes)

    for name in classes:
        full_dir = Path.cwd()/Path(oldpath)/Path(name)
        files = [i.name for i in full_dir.iterdir()]
        total_file = len(files)

        # We split data set into 3: train, validation and test
        
        train_size = math.ceil(total_file * 3/4) # 75% for training 

        validation_size = train_size + math.ceil(total_file * 1/8) # 12.5% for validation
        test_size = validation_size + math.ceil(total_file * 1/8) # 12.5x% for testing 
        
        train = files[0:train_size]
        validation = files[train_size:validation_size]
        test = files[validation_size:]

        move_files(train, full_dir, f"train/{name}")
        move_files(validation, full_dir, f"validation/{name}")
        move_files(test, full_dir, f"test/{name}")

def move_files(files, old_dir, new_dir):
    new_dir = Path.cwd()/Path(new_dir)

    if not new_dir.exists():
        new_dir.mkdir(parents=True)

    for file in files:
        old_file_path = old_dir/file
        new_file_path = new_dir/file
        old_file_path.replace(new_file_path)

def main():
    split_files('101_ObjectCategories', './', classes=None)
    print('Split done')

if __name__ == '__main__':
    main()