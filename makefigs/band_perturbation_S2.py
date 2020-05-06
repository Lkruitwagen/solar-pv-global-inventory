import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.style.use('ggplot')


gg_colors = {
    'red':'#E24A33',
    'blue':'#348ABD',
    'purple':'#988ED5',
    'gray':'#777777',
    'orange':'#FBC15E',
    'green':'#8EBA42',
    'pink':'#FFB5B8'

}

print ([ii['color'] for ii in list(plt.rcParams['axes.prop_cycle'])])

def hex2rgb(h):
    h = h.lstrip('#')
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

df = pd.read_csv(os.path.join(os.getcwd(),'data','band_perturbation','band_perturbation_S2.csv')).sort_values('records').set_index(['records','bands'])

idx = pd.IndexSlice

all_bands = list(df.index.get_level_values(1).unique())
all_bands = [bb for bb in all_bands if bb!='none']

all_bands = [{'name': 'coastal-aerosol', 'resolution':'60m', 'color':'purple' },
             {'name': 'blue',            'resolution':'10m', 'color':'blue'    },
             {'name': 'green',           'resolution':'10m', 'color':'green' },
             {'name': 'red',             'resolution':'10m', 'color':'red' },
             {'name': 'red-edge',        'resolution':'20m', 'color':'pink' },
             {'name': 'red-edge-2',      'resolution':'20m', 'color':'pink' },
             {'name': 'red-edge-3',      'resolution':'20m', 'color':'pink'  },
             {'name': 'nir',             'resolution':'10m', 'color':'orange' },
             {'name': 'red-edge-4',      'resolution':'20m', 'color':'pink'  },
             {'name': 'water-vapor',     'resolution':'60m', 'color':'blue' },
             {'name': 'cirrus',          'resolution':'60m', 'color':'green' },
             {'name': 'swir2',           'resolution':'20m', 'color':'orange' },
             {'name': 'swir1',           'resolution':'20m', 'color':'orange' },
             {'name': 'alpha',           'resolution':'10m', 'color':'gray' },
             ]
print (all_bands)

keep_records = (df.loc[idx[:,'none'],'band_dropout']>0).values
df = df[np.repeat(keep_records,15)]

for band in all_bands:
    # impairment
    df.loc[idx[:,band['name']],:] = (df.loc[idx[:,'none'],:].values - df.loc[idx[:,band['name']],:].values) #/ df.loc[idx[:,'none'],:].values

print (df)

fig, axs = plt.subplots(3,1, figsize=(20,10), sharey=True, sharex=True)

# do band dropout
data = []
for ii_b, band in enumerate(all_bands):
    data.append(df.loc[idx[:,band['name']],'band_dropout'].values)

bplot0 = axs[0].boxplot(data, whis='range',patch_artist=True, medianprops = dict(linestyle='-', linewidth=1, color='firebrick'))
axs[0].set_ylabel('IoU Impairment')
axs[0].set_title('Band Dropout')
axs[0].yaxis.set_major_formatter(mtick.PercentFormatter())

# do additive 1,2,3
data = []
positions=[]
for ii_b, band in enumerate(all_bands):
    data.append(df.loc[idx[:,band['name']],'additive_0.1'].values)
    data.append(df.loc[idx[:,band['name']],'additive_0.2'].values)
    data.append(df.loc[idx[:,band['name']],'additive_0.3'].values)
    positions += [(ii_b+1 + (ii_p-1)/4) for ii_p in range(3)]

bplot1 = axs[1].boxplot(data, positions=positions, whis='range', widths=0.15, patch_artist=True, medianprops = dict(linestyle='-', linewidth=1, color='firebrick'))
axs[1].set_ylabel('IoU Impairment')
axs[1].set_title('Additive Noise [10%, 20%, 30%]')
axs[1].yaxis.set_major_formatter(mtick.PercentFormatter())


# do multiplicative 1,2,3
data = []
positions=[]
for ii_b, band in enumerate(all_bands):
    data.append(df.loc[idx[:,band['name']],'multiplicative_0.1'].values)
    data.append(df.loc[idx[:,band['name']],'multiplicative_0.2'].values)
    data.append(df.loc[idx[:,band['name']],'multiplicative_0.3'].values)
    positions += [(ii_b+1 + (ii_p-1)/4) for ii_p in range(3)]

bplot2 = axs[2].boxplot(data, positions=positions, whis='range', widths=0.15,patch_artist=True, medianprops = dict(linestyle='-', linewidth=1, color='firebrick'))
axs[2].set_xticks(range(1,15))
axs[2].set_xticklabels(['{}$_{{'.format(band['name'])+'{}}}$'.format(band['resolution']) for band in all_bands])
axs[2].set_ylabel('IoU Impairment')
axs[2].set_title('Multiplicative Noise [10%, 20%, 30%]')
axs[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.))

#for bplot in (bplot1, bplot2):
for patch, band in zip(bplot0['boxes'], all_bands):
    #print (dir(patch))
    patch.set_facecolor(gg_colors[band['color']])

for patch, band in zip(bplot1['boxes'], np.repeat(all_bands,3)):
    #print (dir(patch))
    patch.set_facecolor(gg_colors[band['color']])

for patch, band in zip(bplot2['boxes'], np.repeat(all_bands,3)):
    #print (dir(patch))
    patch.set_facecolor(gg_colors[band['color']])

fig.savefig('./band_perturbation.png')
plt.show()