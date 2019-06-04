# dims(.shape) - (600, 32, 32, 3)
# dims for mask?
# outputPath - where we store the hdf5 file
# datakey - name of the dataset - just for indication
# buffSize - size of in-memory buffer (1000 images/feature vectors)
import os
import h5py

class data_to_hdf5:
    def __init__(self, img_dims, label_dims, outputPath, dataKey="images", buffSize=1000):
        # value error can be raised here if path already exists
        if os.path.exists(outputPath):
            pass
        # open HDF5 database for writing("w")
        # create two datasets - images, labels(masks)
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, img_dims, dtype="float")
        # label dims might need to be changed (from rgb(3) to num_classes) + dtype?
        self.labels = self.db.create_dataset("labels", label_dims, dtype="float")

        self.buffSize = buffSize
        self.buffer ={"data": [], "labels": []}
        self.idx = 0

    #'add' function adding data to buffer extend(): Extends list by appending elements from the iterable.
    def add(self, rows, labels):
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels) # rows - labels?

    def flush(self):
        i = self.idx + len(self.buffer["data"]) # determine next available row
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"] # apply numpy slicing to store data+label
        self.idx = i
        self.buffer = {"data":[], "labels":[]} # buffer reset

    # check for remaining data + close
    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()
            self.db.close()
