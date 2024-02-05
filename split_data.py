import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from ProposedAlgorithm.KernelOversampling import KernelOversampling
from sklearn.utils import shuffle
from collections import Counter

filename = 'jdt'
df = pd.read_csv(f'input/{filename}.csv')
df.sort_values(by='commitdate', ascending=True, inplace=True, )
df = df.reset_index(drop=True)
df = df.drop(['transactionid', 'commitdate', 'nm', 'rexp'], axis=1)  # drop commitId
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

indices = np.arange(len(X))
X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2,
                                                                                 random_state=1)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
Y_train = Y_train.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)



# Oversampling
norm = StandardScaler(with_mean=False)
norm_Xtrain = norm.fit_transform(X_train)

oversampler = KernelOversampling()
X_resample, y_resample = oversampler.fit_sample(norm_Xtrain, Y_train)

# X_resample, Y_resample = oversampler.fit_sample(norm_Xtrain, Y_train[:index])
#
# norm_Xtrain = norm.fit_transform(secondHalf)
# secondX_resample, secondY_resample = oversampler.fit_sample(norm_Xtrain, Y_train[index:second])
#
# append_xTrain, append_yTrain = np.append(X_resample,secondX_resample, axis = 0), np.append(Y_resample, secondY_resample,axis = 0)
#
# norm_Xtrain = norm.fit_transform(thirdHalf)
# thirdX_resample, thirdY_resample = oversampler.fit_sample(norm_Xtrain, Y_train[second:])
#
# append_xTrain, append_yTrain = np.append(append_xTrain,thirdX_resample, axis = 0), np.append(append_yTrain, thirdY_resample,axis = 0)


X_shuffle, Y_shuffle = shuffle(X_resample, y_resample)



# X_shuffle, Y_shuffle = shuffle(X_resample, Y_resample)


X_train.to_csv(f'input/split_80_{filename}/X_train.csv', index=False)
X_test.to_csv(f'input/split_80_{filename}/X_test.csv', index=False)
Y_test.to_csv(f'input/split_80_{filename}/Y_test.csv', index=False)
Y_train.to_csv(f'input/split_80_{filename}/Y_train.csv', index=False)
np.savetxt(f'input/split_80_{filename}/indices_test.csv', indices_test, delimiter=',')
np.savetxt(f'input/split_80_{filename}/indices_train.csv', indices_train, delimiter=',')

np.savetxt(f'input/split_80_{filename}/Yresample.csv', Y_shuffle, delimiter=',')
np.save(f'input/split_80_{filename}/Xresample.npy', X_shuffle)


