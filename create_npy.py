import numpy as np

csvfile = '/data/xjiang/fishing/mobility_bin_50.csv'
task = 'mobility_bin_50'

data = np.genfromtxt(csvfile, dtype=float, delimiter=',')
mmsi = data[:,0].astype(int)
labels = data[:,data.shape[1]-1]
features = data[:, 1:data.shape[1]-1]

mmsi_length = np.unique(mmsi, return_counts=True)
num_vessels = len(mmsi_length[0])
max_mmsi_len = max(mmsi_length[1])
num_features = features.shape[1]

padding_data = np.zeros((max_mmsi_len * num_vessels, num_features))
for vessel in range(num_vessels):
    vessel_data = np.zeros((max_mmsi_len, num_features))
    vessel_data[0:mmsi_length[1][vessel],:] = features[mmsi == mmsi_length[0][vessel],:]
    padding_data[vessel*max_mmsi_len : (vessel+1)*max_mmsi_len,:] = vessel_data
    

padding_labels = np.zeros((max_mmsi_len * num_vessels))
for vessel in range(num_vessels):
    vessel_data = np.zeros((max_mmsi_len))
    vessel_data[0:mmsi_length[1][vessel]] = labels[mmsi == mmsi_length[0][vessel]]
    padding_labels[vessel*max_mmsi_len : (vessel+1)*max_mmsi_len] = vessel_data
    


x = np.reshape(padding_data, (num_vessels, max_mmsi_len, num_features))
y = np.reshape(padding_labels, (num_vessels, max_mmsi_len))
y = y.astype(int)
np.save('/data/xjiang/fishing/x_'+task+'.npy', x)
np.save('/data/xjiang/fishing/y_'+task+'.npy', y)
np.save('/data/xjiang/fishing/mmsi_'+task+'.npy', mmsi_length)


