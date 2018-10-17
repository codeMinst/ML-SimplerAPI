import os
import numpy as np
import pickle
'''
DATA_XY_FILE = 'dataXY.npz'

if os.path.isfile(DATA_XY_FILE):
    dataXY = np.load(DATA_XY_FILE)
    xx, yy = dataXY['x'], dataXY['y']
    print("hi module")
'''

with open('hs_svm_work_unknown.pkl', 'rb') as f:
    svm = pickle.load(f)
tt = [[-0.56010646]]
result = svm.predict(tt)
print(result)


def testPrint():
    print("hi module")