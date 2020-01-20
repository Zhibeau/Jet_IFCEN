import numpy as np
import pandas as pd
from pandas import DataFrame,Series

jet = pd.read_csv('/opt/data/Particle_Classification/jet_complex_data/complex_train_R04_jet.csv')

particle = pd.read_csv('/opt/data/Particle_Classification/jet_complex_data/complex_train_R04_particle.csv')

event = pd.read_csv('/opt/data/Particle_Classification/jet_complex_data/complex_train_R04_event.csv')

for ids in event.event_id:
    jet_ids=jet.label[jet['event_id'].isin([ids])]
    for i in range(0,len(jet_ids)-2):
        if int(jet_ids[i:i+1])!=int(jet_ids[i+1:i+2]):
            print(jet_ids)