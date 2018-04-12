from matplotlib import pyplot as plt
import pickle
import numpy as np

# USING MEAN METHOD#
mean_chaise = []
mean_chaise.append(pickle.load(open('../recordings/1st take/results/CHAISE_COUTURIER_ELODIE_HR_MEAN_AE.p', 'rb')))
mean_chaise.append(pickle.load(open('../recordings/1st take/results/CHAISE_LEMAY_RAPHAEL_HR_MEAN_AE.p', 'rb')))
mean_chaise.append(pickle.load(open('../recordings/1st take/results/CHAISE_OTIS_SAMUEL_HR_MEAN_AE.p', 'rb')))
mean_chaise.append(pickle.load(open('../recordings/1st take/results/CHAISE_LEMIEUX_NICOLAS_HR_MEAN_AE.p', 'rb')))
# plt.figure()
# plt.title('Mean_chaise')
# plt.boxplot(mean_chaise)
# plt.pause(0.0001)

mean_sol = []
mean_sol.append(pickle.load(open('../recordings/1st take/results/SOL_COUTURIER_ELODIE_HR_MEAN_AE.p', 'rb')))
mean_sol.append(pickle.load(open('../recordings/1st take/results/SOL_LEMAY_RAPHAEL_HR_MEAN_AE.p', 'rb')))
mean_sol.append(pickle.load(open('../recordings/1st take/results/SOL_OTIS_SAMUEL_HR_MEAN_AE.p', 'rb')))
mean_sol.append(pickle.load(open('../recordings/1st take/results/SOL_LEMIEUX_NICOLAS_HR_MEAN_AE.p', 'rb')))
# plt.figure()
# plt.title('Mean_sol')
# plt.boxplot(mean_sol)
# plt.pause(0.0001)

mean_mae = [np.mean(mean_i) for mean_i in mean_chaise + mean_sol]
mean_amae = np.mean(mean_mae)
mean_std = np.std(mean_mae)

# USING Euclidean distance METHOD#
e_dis_chaise = []
e_dis_sol = []

e_dis_chaise.append(pickle.load(open('../recordings/1st take/results/CHAISE_COUTURIER_ELODIE_HR_DIS_AE.p', 'rb')))
e_dis_chaise.append(pickle.load(open('../recordings/1st take/results/CHAISE_LEMAY_RAPHAEL_HR_DIS_AE.p', 'rb')))
e_dis_chaise.append(pickle.load(open('../recordings/1st take/results/CHAISE_OTIS_SAMUEL_HR_DIS_AE.p', 'rb')))
e_dis_chaise.append(pickle.load(open('../recordings/1st take/results/CHAISE_LEMIEUX_NICOLAS_HR_DIS_AE.p', 'rb')))
# plt.figure()
# plt.title('Euclidean_chaise')
# plt.boxplot(e_dis_chaise)
# plt.pause(0.0001)


e_dis_sol.append(pickle.load(open('../recordings/1st take/results/SOL_COUTURIER_ELODIE_HR_DIS_AE.p', 'rb')))
e_dis_sol.append(pickle.load(open('../recordings/1st take/results/SOL_LEMAY_RAPHAEL_HR_DIS_AE.p', 'rb')))
e_dis_sol.append(pickle.load(open('../recordings/1st take/results/SOL_OTIS_SAMUEL_HR_DIS_AE.p', 'rb')))
e_dis_sol.append(pickle.load(open('../recordings/1st take/results/SOL_LEMIEUX_NICOLAS_HR_DIS_AE.p', 'rb')))
# plt.figure()
# plt.title('Euclidean_sol')
# plt.boxplot(e_dis_sol)
# plt.pause(0.0001)

e_dis_mae = [np.mean(e_dis_i) for e_dis_i in e_dis_chaise + e_dis_sol]
e_dis_amae = np.mean(e_dis_mae)
e_dis_std = np.std(e_dis_mae)

# USING EIGEN VALUES METHOD#
eigen_chaise = []
eigen_sol = []

eigen_chaise.append(pickle.load(open('../recordings/2nd take/results/CHAISE_COUTURIER_ELODIE_HRZ_EIGEN_AE.p', 'rb')))
eigen_chaise.append(pickle.load(open('../recordings/2nd take/results/CHAISE_LEMAY_RAPHAEL_HRZ_EIGEN_AE.p', 'rb')))
eigen_chaise.append(pickle.load(open('../recordings/2nd take/results/CHAISE_OTIS_SAMUEL_HRZ_EIGEN_AE.p', 'rb')))
plt.figure()
plt.title('Eigen_chaise')
plt.boxplot(eigen_chaise)
plt.pause(0.0001)

eigen_sol.append(pickle.load(open('../recordings/2nd take/results/SOL_COUTURIER_HRZ_EIGEN_AE.p', 'rb')))
eigen_sol.append(pickle.load(open('../recordings/2nd take/results/SOL_LEMAY_RAPHAEL_HRZ_EIGEN_AE.p', 'rb')))
eigen_sol.append(pickle.load(open('../recordings/2nd take/results/SOL_OTIS_SAMUEL_HRZ_EIGEN_AE.p', 'rb')))
plt.figure()
plt.title('Eigen_sol')
plt.boxplot(eigen_sol)
plt.pause(0.0001)
eigen_mae = [np.mean(eigen_i) for eigen_i in eigen_chaise + eigen_sol]
eigen_amae_mean = np.mean(eigen_mae)
eigen_std = np.std(eigen_mae)

# Mean vs Eigen comparisons#
plt.figure()
plt.title('Méthodes comparées')
plt.boxplot([mean_mae, e_dis_mae, eigen_mae])
plt.xticks([1, 2, 3], ['Moyenne', 'Euclid', 'Eigen'])
plt.xlabel('Méthodes')
plt.ylabel('AMAE (BPM)')
plt.pause(0.00001)

# Chaise vs Sol#
chaise_mae = [np.mean(i) for i in mean_chaise + e_dis_chaise + eigen_chaise]
sol_mae = [np.mean(i) for i in mean_sol + e_dis_sol + eigen_sol]
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
