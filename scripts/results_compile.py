from matplotlib import pyplot as plt
import pickle
import numpy as np
import os
# USING EIGEN VALUES x METHOD#
mypath = '../recordings/2nd take/pcl_eigenvalues/results/db6/euclidean/'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
eigen_chaise_euc = [pickle.load(open(mypath + file, 'rb')) for file in files if 'CHAISE' in file]
# plt.figure()
# plt.title('Eigen_chaise')
# plt.boxplot(eigen_chaise)
# plt.pause(0.0001)

eigen_sol_euc = [pickle.load(open(mypath + file, 'rb')) for file in files if 'SOL' in file]
# plt.figure()
# plt.title('Eigen_sol')
# plt.boxplot(eigen_sol)
# plt.pause(0.0001)
eigen_euc_mae = [np.mean(i) for i in eigen_chaise_euc + eigen_sol_euc]
eigen_euc_amae_mean = np.mean(eigen_euc_mae)
eigen_euc_std = np.std(eigen_euc_mae)

# USING EIGEN VALUES x METHOD#
mypath = '../recordings/2nd take/pcl_eigenvalues/results/db6/mean_xyz/'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
eigen_chaise_mean = [pickle.load(open(mypath + file, 'rb')) for file in files if 'CHAISE' in file]
# plt.figure()
# plt.title('Eigen_chaise')
# plt.boxplot(eigen_chaise)
# plt.pause(0.0001)

eigen_sol_mean = [pickle.load(open(mypath + file, 'rb')) for file in files if 'SOL' in file]
# plt.figure()
# plt.title('Eigen_sol')
# plt.boxplot(eigen_sol)
# plt.pause(0.0001)
eigen_mean_mae = [np.mean(i) for i in eigen_chaise_mean + eigen_sol_mean]
eigen_mean_amae_mean = np.mean(eigen_mean_mae)
eigen_mean_std = np.std(eigen_mean_mae)

# USING EIGEN VALUES x METHOD#
mypath = '../recordings/2nd take/pcl_eigenvalues/results/db6/x/'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
eigen_chaise_x = [pickle.load(open(mypath + file, 'rb')) for file in files if 'CHAISE' in file]
# plt.figure()
# plt.title('Eigen_chaise')
# plt.boxplot(eigen_chaise)
# plt.pause(0.0001)

eigen_sol_x = [pickle.load(open(mypath + file, 'rb')) for file in files if 'SOL' in file]
# plt.figure()
# plt.title('Eigen_sol')
# plt.boxplot(eigen_sol)
# plt.pause(0.0001)
eigen_x_mae = [np.mean(i) for i in eigen_chaise_x + eigen_sol_x]
eigen_x_amae_mean = np.mean(eigen_x_mae)
eigen_x_std = np.std(eigen_x_mae)

# USING EIGEN VALUES y METHOD#
mypath = '../recordings/2nd take/pcl_eigenvalues/results/db6/y/'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
eigen_chaise_y = [pickle.load(open(mypath + file, 'rb')) for file in files if 'CHAISE' in file]
# plt.figure()
# plt.title('Eigen_chaise')
# plt.boxplot(eigen_chaise)
# plt.pause(0.0001)

eigen_sol_y = [pickle.load(open(mypath + file, 'rb')) for file in files if 'SOL' in file]
# plt.figure()
# plt.title('Eigen_sol')
# plt.boxplot(eigen_sol)
# plt.pause(0.0001)
eigen_y_mae = [np.mean(i) for i in eigen_chaise_y + eigen_sol_y]
eigen_y_amae_mean = np.mean(eigen_y_mae)
eigen_y_std = np.std(eigen_y_mae)


# USING EIGEN VALUES z METHOD#
mypath = '../recordings/2nd take/pcl_eigenvalues/results/db6/z/'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
eigen_chaise_z = [pickle.load(open(mypath + file, 'rb')) for file in files if 'CHAISE' in file]
# plt.figure()
# plt.title('Eigen_chaise z')
# plt.boxplot(eigen_chaise)
# plt.pause(0.0001)

eigen_sol_z = [pickle.load(open(mypath + file, 'rb')) for file in files if 'SOL' in file]
# plt.figure()
# plt.title('Eigen_sol')
# plt.boxplot(eigen_sol)
# plt.pause(0.0001)
eigen_z_mae = [np.mean(i) for i in eigen_chaise_z + eigen_sol_z]
eigen_z_amae_mean = np.mean(eigen_z_mae)
eigen_z_std = np.std(eigen_z_mae)
#############################################################################################################

# USING centroid mean VALUES METHOD#
mypath = '../recordings/2nd take/centroids/results/db6/meanxyz/'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

centroid_meanxyz_chaise = [pickle.load(open(mypath + file, 'rb')) for file in files if 'CHAISE' in file]
# plt.figure()
# plt.title('Centroid_mean_chaise')
# plt.boxplot(centroid_meanxyz_chaise)
# plt.pause(0.0001)

centroid_meanxyz_sol = [pickle.load(open(mypath + file, 'rb')) for file in files if 'SOL' in file]
# plt.figure()
# plt.title('Centroid_mean_sol')
# plt.boxplot(centroid_meanxyz_sol)
# plt.pause(0.0001)

centroid_meanxyz_mae = [np.mean(i) for i in centroid_meanxyz_chaise + centroid_meanxyz_sol]
centroid_meanxyz_amae_mean = np.mean(centroid_meanxyz_mae)
centroid_meanxyz_std = np.std(centroid_meanxyz_mae)

#############################################################################################################

# USING centroid euclidean distance VALUES METHOD#
mypath = '../recordings/2nd take/centroids/results/db6/euclidean/'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

centroid_euc_chaise = [pickle.load(open(mypath + file, 'rb')) for file in files if 'CHAISE' in file]
# plt.figure()
# plt.title('Centroid_euc_chaise')
# plt.boxplot(centroid_euc_chaise)
# plt.pause(0.0001)

centroid_euc_sol = [pickle.load(open(mypath + file, 'rb')) for file in files if 'SOL' in file]
# plt.figure()
# plt.title('Centroid_euc_sol')
# plt.boxplot(centroid_euc_sol)
# plt.pause(0.0001)

centroid_euc_mae = [np.mean(i) for i in centroid_euc_chaise + centroid_euc_sol]
centroid_euc_amae_mean = np.mean(centroid_euc_mae)
centroid_euc_std = np.std(centroid_euc_mae)
#############################################################################################################

# USING centroid x value METHOD#
mypath = '../recordings/2nd take/centroids/results/db6/x/'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

centroid_x_chaise = [pickle.load(open(mypath + file, 'rb')) for file in files if 'CHAISE' in file]
# plt.figure()
# plt.title('Centroid_x_chaise')
# plt.boxplot(centroid_x_chaise)
# plt.pause(0.0001)

centroid_x_sol = [pickle.load(open(mypath + file, 'rb')) for file in files if 'SOL' in file]
# plt.figure()
# plt.title('Centroid_x_sol')
# plt.boxplot(centroid_x_sol)
# plt.pause(0.0001)

centroid_x_mae = [np.mean(i) for i in centroid_x_chaise + centroid_x_sol]
centroid_x_amae_mean = np.mean(centroid_x_mae)
centroid_x_std = np.std(centroid_x_mae)
#######################################################################################
# USING centroid y value VALUES METHOD#
mypath = '../recordings/2nd take/centroids/results/db6/y/'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

centroid_y_chaise = [pickle.load(open(mypath + file, 'rb')) for file in files if 'CHAISE' in file]
# plt.figure()
# plt.title('Centroid_y_chaise')
# plt.boxplot(centroid_y_chaise)
# plt.pause(0.0001)

centroid_y_sol = [pickle.load(open(mypath + file, 'rb')) for file in files if 'SOL' in file]
# plt.figure()
# plt.title('Centroid_y_sol')
# plt.boxplot(centroid_y_sol)
# plt.pause(0.0001)

centroid_y_mae = [np.mean(i) for i in centroid_y_chaise + centroid_y_sol]
centroid_y_amae_mean = np.mean(centroid_y_mae)
centroid_y_std = np.std(centroid_y_mae)
#############################################################################################################

# USING centroid z value VALUES METHOD#
mypath = '../recordings/2nd take/centroids/results/db6/z/'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

centroid_z_chaise = [pickle.load(open(mypath + file, 'rb')) for file in files if 'CHAISE' in file]
# plt.figure()
# plt.title('Centroid_z_chaise')
# plt.boxplot(centroid_z_chaise)
# plt.pause(0.0001)

centroid_z_sol = [pickle.load(open(mypath + file, 'rb')) for file in files if 'SOL' in file]
# plt.figure()
# plt.title('Centroid_z_sol')
# plt.boxplot(centroid_z_sol)
# plt.pause(0.0001)

centroid_z_mae = [np.mean(i) for i in centroid_z_chaise + centroid_z_sol]
centroid_z_amae_mean = np.mean(centroid_z_mae)
centroid_z_std = np.std(centroid_z_mae)
#############################################################################################################
#################################################################################3
###################################################################################

# USING mean mean VALUES METHOD#
mypath = '../recordings/2nd take/mean_xyz/results/db6/mean_xyz/'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

mean_meanxyz_chaise = [pickle.load(open(mypath + file, 'rb')) for file in files if 'CHAISE' in file]
# plt.figure()
# plt.title('Mean_mean_chaise mean xyz')
# plt.boxplot(mean_meanxyz_chaise)
# plt.pause(0.0001)

mean_meanxyz_sol = [pickle.load(open(mypath + file, 'rb')) for file in files if 'SOL' in file]
# plt.figure()
# plt.title('Mean_mean_sol mean xyz')
# plt.boxplot(mean_meanxyz_sol)
# plt.pause(0.0001)

mean_meanxyz_mae = [np.mean(i) for i in mean_meanxyz_chaise + mean_meanxyz_sol]
mean_meanxyz_amae_mean = np.mean(mean_meanxyz_mae)
mean_meanxyz_std = np.std(mean_meanxyz_mae)

#############################################################################################################

# USING mean euclidean distance VALUES METHOD#
mypath = '../recordings/2nd take/mean_xyz/results/db6/euclidean/'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

mean_euc_chaise = [pickle.load(open(mypath + file, 'rb')) for file in files if 'CHAISE' in file]
# plt.figure()
# plt.title('Mean_euc_chaise euclidean')
# plt.boxplot(mean_euc_chaise)
# plt.pause(0.0001)

mean_euc_sol = [pickle.load(open(mypath + file, 'rb')) for file in files if 'SOL' in file]
# plt.figure()
# plt.title('Mean_euc_sol euclidean')
# plt.boxplot(mean_euc_sol)
# plt.pause(0.0001)

mean_euc_mae = [np.mean(eigen_i) for eigen_i in mean_euc_chaise + mean_euc_sol]
mean_euc_amae_mean = np.mean(mean_euc_mae)
mean_euc_std = np.std(mean_euc_mae)
#############################################################################################################

# USING mean x value METHOD#
mypath = '../recordings/2nd take/mean_xyz/results/db6/x/'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

mean_x_chaise = [pickle.load(open(mypath + file, 'rb')) for file in files if 'CHAISE' in file]
# plt.figure()
# plt.title('Mean_euc_chaise on x')
# plt.boxplot(mean_x_chaise)
# plt.pause(0.0001)

mean_x_sol = [pickle.load(open(mypath + file, 'rb')) for file in files if 'SOL' in file]
plt.figure()
plt.title('Centroid_euc_sol on x')
plt.boxplot(mean_x_sol)
plt.pause(0.0001)

mean_x_mae = [np.mean(eigen_i) for eigen_i in mean_x_chaise + mean_x_sol]
mean_x_amae_mean = np.mean(mean_x_mae)
mean_x_std = np.std(mean_x_mae)
#######################################################################################
# USING mean y value VALUES METHOD#
mypath = '../recordings/2nd take/mean_xyz/results/db6/y/'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

mean_y_chaise = [pickle.load(open(mypath + file, 'rb')) for file in files if 'CHAISE' in file]
# plt.figure()
# plt.title('mean_y_chaise on y')
# plt.boxplot(mean_y_chaise)
# plt.pause(0.0001)

mean_y_sol = [pickle.load(open(mypath + file, 'rb')) for file in files if 'SOL' in file]
# plt.figure()
# plt.title('mean_y_sol on y')
# plt.boxplot(mean_y_sol)
# plt.pause(0.0001)

mean_y_mae = [np.mean(i) for i in mean_y_chaise + mean_y_sol]
mean_y_amae_mean = np.mean(mean_y_mae)
mean_y_std = np.std(mean_y_mae)
#############################################################################################################

# USING mean z value VALUES METHOD#
mypath = '../recordings/2nd take/mean_xyz/results/db6/z/'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

mean_z_chaise = [pickle.load(open(mypath + file, 'rb')) for file in files if 'CHAISE' in file]
# plt.figure()
# plt.title('mean_z_chaise on z')
# plt.boxplot(mean_z_chaise)
# plt.pause(0.0001)

mean_z_sol = [pickle.load(open(mypath + file, 'rb')) for file in files if 'SOL' in file]
# plt.figure()
# plt.title('mean_z_sol on z')
# plt.boxplot(mean_z_sol)
# plt.pause(0.0001)

mean_z_mae = [np.mean(i) for i in mean_z_chaise + mean_z_sol]
mean_z_amae_mean = np.mean(mean_z_mae)
mean_z_std = np.std(mean_z_mae)
#############################################################################################################

# Mean vs Eigen comparisons#
plt.figure()
plt.title('Méthodes comparées')
plt.boxplot([eigen_euc_mae, eigen_mean_mae, eigen_x_mae, eigen_y_mae, eigen_z_mae, centroid_meanxyz_mae, centroid_euc_mae, centroid_x_mae, centroid_y_mae,
             centroid_z_mae, mean_meanxyz_mae, mean_euc_mae, mean_x_mae, mean_y_mae, mean_z_mae])
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 15],
           ['Eigen_eux', 'Eigen_mean', 'Eigen_x', 'Eigen_y', 'Eigen_z', 'Centroid_xyz', 'Centroid_euc', 'Centr x', 'Centr y', 'Centr z', 'Mean_xyz', 'Mean_euc', 'Mean x', 'Mean y', 'Mean z'])
plt.xlabel('Méthodes')
plt.ylabel('AMAE (BPM)')
plt.pause(0.00001)

# Chaise vs Sol#
chaise_mae = [np.mean(i) for i in eigen_chaise_euc + eigen_chaise_mean + eigen_chaise_x + eigen_chaise_y + eigen_chaise_z + centroid_euc_chaise + centroid_meanxyz_chaise + centroid_x_chaise + centroid_y_chaise + centroid_z_chaise]
sol_mae = [np.mean(i) for i in eigen_mean_mae + eigen_x_mae + eigen_y_mae + eigen_z_mae + eigen_euc_mae + centroid_euc_sol + centroid_meanxyz_sol + centroid_x_sol + centroid_y_sol + centroid_z_sol]
chaise_amae = np.mean(chaise_mae)
chaise_std = np.std(chaise_mae)
sol_amae = np.mean(sol_mae)
sol_std = np.std(sol_mae)
plt.figure()
plt.title('Positionnement comparé')
plt.boxplot([chaise_mae, sol_mae])
plt.xlabel('Positionnement')
plt.ylabel('AMAE (BPM)')
plt.xticks([1, 2], ['Chaise', 'Au sol'])
plt.pause(0.00001)
assert True
