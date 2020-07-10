import random as r
import os
import pandas as pd
import time

def generate_partitions():
    df = pd.read_csv("./train.csv")
    print(df.head(5))
    mapping = {k: v for v, k in enumerate(df.diagnosis.unique())}
    df['diagnosis_e'] = df.diagnosis.map(mapping)

    mapping2 = {k: v for v, k in enumerate(df.anatom_site_general_challenge.unique())}
    df['anatom_site_general_challenge_e'] = df.anatom_site_general_challenge.map(mapping2)

    is_female =  df['sex']=='female'
    is_male = df['sex'] == 'male'
    is_ill = df['target'] == 1
    is_not_ill = df['target'] == 0
    print(df[is_ill & is_female].shape)
    print(df[is_ill & is_male].shape)
    print(df[is_ill ].shape)

    ifv_value_count = {6: 1,
                       4: 1,
                       3: 15,
                       2: 10,
                       1: 10,
                       0: 5
                       }

    imv_value_count = {5: 2,
                       4: 2,
                       3: 30,
                       2: 12,
                       1: 12,
                       0: 10
                       }

    cgv_value_count = {6: 5,
                       5: 5,
                       4: 5,
                       3: 100,
                       2: 50,
                       1: 50,
                       0: 16
                       }

    cgt_value_count = {6: 25,
                       5: 25,
                       4: 25,
                       3: 500,
                       2: 250,
                       1: 250,
                       0: 80
                       }

    def target_indices(f, value_count):
        indices = []
        for index, row in f.iterrows():
            for key in value_count:
                if key == row['anatom_site_general_challenge_e'] and value_count[key] > 0:
                    indices.append(index)
                    value_count[key] -= 1
        return (indices)


    print(df.diagnosis.unique())
    print(df[is_ill & is_male]['anatom_site_general_challenge_e'].value_counts())
    print("mmmmmmmmmmmmmmm")
    print(df[is_ill & is_female]['anatom_site_general_challenge_e'].value_counts())

    def restore(f) :
        fn = "./tmp_"+str( r.randrange(100)  )+".csv"
        f.to_csv(fn)
        time.sleep(1)
        of = pd.read_csv(fn)
        #os.remove(fn)
        return of
    df = pd.read_csv("./train.csv")
    mapping2 = {k: v for v, k in enumerate(df.anatom_site_general_challenge.unique())}
    df['anatom_site_general_challenge_e'] = df.anatom_site_general_challenge.map(mapping2)
    df = df.sample(frac=1).reset_index(drop=True)
    is_female = df['sex'] == 'female'
    is_male = df['sex'] == 'male'
    is_ill = df['target'] == 1
    is_not_ill = df['target'] == 0

    dfif = restore(df[is_ill & is_female])

    indices = target_indices(dfif, ifv_value_count)
    print(indices)
    dfift = dfif.drop(indices).copy()
    print(dfift.shape)
    dfim = restore(df[is_ill & is_male])
    indices = target_indices(dfim, imv_value_count)
    dfimt = dfim.drop(indices).copy()
    print(dfimt.shape)

    dfcm = restore(df[is_not_ill & is_male])
    indices = target_indices(dfcm, cgt_value_count)
    dfcmtt = dfcm.iloc[indices].copy()
    print(dfcmtt.shape)
    cgt_value_count = {6: 25,
                       5: 25,
                       4: 25,
                       3: 500,
                       2: 250,
                       1: 250,
                       0: 80
                       }
    dfcf = restore(df[is_not_ill & is_female])
    indices = target_indices(dfcf, cgt_value_count)
    dfcft = dfcf.iloc[indices].copy()

    train_frames = [ dfimt, dfift, dfcmtt, dfcft ]

    train = pd.concat(train_frames, sort=True)
    print(train.shape)

    train.to_csv("./filtered_train.csv")

    df = pd.read_csv("./train.csv")
    mapping2 = {k: v for v, k in enumerate(df.anatom_site_general_challenge.unique())}
    df['anatom_site_general_challenge_e'] = df.anatom_site_general_challenge.map(mapping2)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    is_female =  df['sex']=='female'
    is_male = df['sex'] == 'male'
    is_ill = df['target'] == 1
    is_not_ill = df['target'] == 0
    cgt_value_count = {6: 25,
                       5: 25,
                       4: 25,
                       3: 500,
                       2: 250,
                       1: 250,
                       0: 80
                       }

    cgv_value_count = {6: 5,
                       5: 5,
                       4: 5,
                       3: 100,
                       2: 50,
                       1: 50,
                       0: 16
                       }
    dmcm = restore(df[is_not_ill & is_male])
    indices = target_indices(dmcm, cgt_value_count)
    dfcmt = dmcm.iloc[indices].copy()

    dfcmtl = restore(dmcm.drop(indices))
    indices = target_indices(dfcmtl, cgv_value_count)
    dfcmv = dfcmtl.iloc[indices].copy()
    cgt_value_count = {6: 25,
                       5: 25,
                       4: 25,
                       3: 500,
                       2: 250,
                       1: 250,
                       0: 80
                       }

    cgv_value_count = {6: 5,
                       5: 5,
                       4: 5,
                       3: 100,
                       2: 50,
                       1: 50,
                       0: 16
                       }

    dfcm = restore(df[is_not_ill & is_female])
    indices = target_indices(dfcm, cgt_value_count)
    dfcft = dfcm.iloc[indices].copy()

    dfcftl = restore(dfcm.drop(indices))
    indices = target_indices(dfcftl, cgv_value_count)
    dfcfv = dfcftl.iloc[indices].copy()

    valid_frames = [restore(df[is_ill]), dfcmt, dfcmv, dfcft,  dfcfv]
    valid = pd.concat(valid_frames, sort=True)
    print(valid.shape)
    valid.to_csv("./filtered_valid.csv")

generate_partitions()