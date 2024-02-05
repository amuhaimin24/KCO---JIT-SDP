from sklearn.decomposition import KernelPCA
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from imblearn.under_sampling import RandomUnderSampler
import janitor as jn



class KernelOversampling(object):
    def __init__(self, pfp=0.5):
        self.data_t = None  # Save the initial defect sample
        self.pfp = pfp  # Proportion of expected defective samples
        self.T = 0  # The number of defect samples that need to be generated
        # self.new = []  # Store the newly generated sample
        self.start = 0.9
        self.jitter = 0.1

    def fit_sample(self, data, label):

        # Undersampling the data to a certain proportion.
        undersampler = RandomUnderSampler(sampling_strategy=self.start)
        data, label = undersampler.fit_resample(data, label)
        new = []  # Store the newly generated sample
        # Doing dimensionality reduction.
        KPCA = KernelPCA(n_components=2, kernel='rbf')
        transformed = KPCA.fit_transform(data)



        # Doing clustering on the transformed data.
        sc = SpectralClustering(n_clusters=10).fit(transformed)


        cluster_labels = sc.labels_
        # Doing clustering on the transformed data.
        dt_KPCA = pd.DataFrame(transformed)
        dt_KPCA.insert(2, 'Labels', cluster_labels, True)
        dt_KPCA['Defect'] = label
        ranking = pd.DataFrame(columns=['Spatial', 'cluster_ID'])

        for i in range(sc.n_clusters):
            n_all = len(dt_KPCA.query(f'Labels == {i}'))
            n_clean = len(dt_KPCA.query(f'Labels == {i} and Defect == 0'))
            spatial_distribution = n_clean / n_all
            ranking = ranking.append({'Spatial': spatial_distribution, 'cluster_ID': i}, ignore_index=True)

        ranking.sort_values('Spatial', inplace=True, ignore_index=True)
        # Only 30% data distribution will be considered because most of the clusters have high spatial distribution
        lowSpatial_clusters = [ranking.iloc[i].cluster_ID for i in range(int(30 / 100 * sc.n_clusters))]

        lowSpatial_KPCA = dt_KPCA.query(f'Labels == {lowSpatial_clusters} and Defect == 1')
        self.data_t = data[lowSpatial_KPCA.index]
        no_rows = len(self.data_t)
        no_cols = data.shape[1]
        firstGen = []
        requirePositive = len(dt_KPCA.query('Defect == 0')) / (1 - self.pfp) - len(dt_KPCA.query('Defect == 0'))
        NumberRequireNew = requirePositive - len(dt_KPCA.query('Defect == 1'))
        # Generate 1st generation of defect data based on crossover interpolation
        for i in range(len(self.data_t)):
            parent_1 = np.random.randint(0, no_rows)
            parent_2 = np.random.randint(0, no_rows)

            # selecting features measurement to be carried out from parent 1 into crossover process
            mask = np.random.randint(0, 2, no_cols)

            new_instance = (self.data_t[parent_1] * mask) + (self.data_t[parent_2] * (1 - mask))
            new.append(new_instance)
            firstGen.append(new_instance)

        currentGen = firstGen

        while len(new) <= NumberRequireNew:

            temp = []
            for i in range(len(self.data_t)):
                parent_1 = np.random.randint(0, len(self.data_t))
                parent_2 = np.random.randint(0, len(currentGen))

                # selecting features measurement to be carried out from parent 1 into crossover process
                mask = np.random.randint(0, 2, no_cols)

                new_instance = self.data_t[parent_1] * mask + currentGen[parent_2] * (1 - mask)
                if len(new) <= NumberRequireNew:
                    new.append(new_instance)
                temp.append(new_instance)
            currentGen = temp

        label_new = np.ones(len(new))
        list_columns = list(range(no_cols))


        dt_jitter = pd.DataFrame(new, columns=list_columns)

        for f in list_columns:
            new_column = 'jitter_' + str(f)
            jn.functions.jitter(df=dt_jitter, column_name=f, dest_column_name=new_column, scale = self.jitter)

            exec(f'dt_jitter.jitter_{f} = ' + f'dt_jitter.jitter_{f}.abs()')

        dt_jitter = dt_jitter.filter(regex='jitter', axis=1)
        new = dt_jitter.values
        return np.append(data, new, axis=0), np.append(label, label_new, axis=0)
