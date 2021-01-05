import scipy.io

def get_dataFile():
    data = scipy.io.loadmat('datasets/data.mat')
    
    train_X = data['X'].T
    train_Y = data['y'].T
    val_X = data['Xval'].T
    val_Y = data['yval'].T

    print(data['X'].T.shape)
    print(data['y'].T.shape)
    print(data['Xval'].T.shape)
    print(data['yval'].T.shape)
    #print('data[\'y\']', data['y'])
    #print('data[\'y\'].T', data['y'].T)
    input()

    return train_X, train_Y, val_X, val_Y
