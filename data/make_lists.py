import os
import pickle
import random


def get_all_file_names(dirName):
    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        # listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        for file in filenames:
            if file.endswith(".jpg"):
                path = os.path.join(os.path.abspath(dirpath), file)
                listOfFiles.append(path.replace("\\", "/"))
    
    return listOfFiles


file_names = get_all_file_names("../formatted_mars_data")
print(file_names)

SPLIT_RATIO = 0.2

random.shuffle(file_names)
split = int(len(file_names) * SPLIT_RATIO)

train = file_names[:len(file_names)-split]
test = file_names[len(file_names)-split:]

with open("train.txt", "wb") as fp:
    pickle.dump(train, fp)

with open("test.txt", "wb") as fp:
    pickle.dump(test, fp)