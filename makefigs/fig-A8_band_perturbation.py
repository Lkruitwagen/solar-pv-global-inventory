import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec

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

df_S2 = pd.read_csv(os.path.join(os.getcwd(),'data','band_perturbation','band_perturbation_S2.csv')).sort_values('records').set_index(['records','bands'])
df_SPOT = pd.read_csv(os.path.join(os.getcwd(),'data','band_perturbation','band_perturbation_SPOT.csv'))


# prep SPOT df




idx = pd.IndexSlice

#all_bands = list(df.index.get_level_values(1).unique())
#all_bands = [bb for bb in all_bands if bb!='none']

S2_bands = [{'name': 'coastal-aerosol', 'resolution':'60m', 'color':'purple' },
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


SPOT_bands = [  {'name':'red',   'resolution':'1.5m',  'color':'red'},
                {'name':'green', 'resolution':'1.5m',  'color':'green'},
                {'name':'blue',  'resolution':'1.5m',  'color':'blue'},
                {'name':'nir',   'resolution':'1.5m',  'color':'orange'},]
print (S2_bands)

keep_records = (df_S2.loc[idx[:,'none'],'band_dropout']>0).values
df_S2 = df_S2[np.repeat(keep_records,15)]
df_SPOT = df_SPOT[df_SPOT['None']>0]

for band in S2_bands:
    # impairment
    df_S2.loc[idx[:,band['name']],:] = (df_S2.loc[idx[:,'none'],:].values - df_S2.loc[idx[:,band['name']],:].values) # / df_S2.loc[idx[:,'none'],:].values

for ii_b, band in enumerate(SPOT_bands):
    df_SPOT[band['name']+'_bdo'] = (df_SPOT['None'] - df_SPOT['bdo_'+str(ii_b)]) #/ df_SPOT['None']
    df_SPOT[band['name']+'_additive_0.1'] = (df_SPOT['None'] - df_SPOT['additive_0.1_'+str(ii_b)]) #/ df_SPOT['None']
    df_SPOT[band['name']+'_additive_0.2'] = (df_SPOT['None'] - df_SPOT['additive_0.2_'+str(ii_b)]) #/ df_SPOT['None']
    df_SPOT[band['name']+'_additive_0.3'] = (df_SPOT['None'] - df_SPOT['additive_0.3_'+str(ii_b)]) #/ df_SPOT['None']
    df_SPOT[band['name']+'_multiplicative_0.1'] = (df_SPOT['None'] - df_SPOT['multiplicative_0.1_'+str(ii_b)]) #/ df_SPOT['None']
    df_SPOT[band['name']+'_multiplicative_0.2'] = (df_SPOT['None'] - df_SPOT['multiplicative_0.2_'+str(ii_b)]) #/ df_SPOT['None']
    df_SPOT[band['name']+'_multiplicative_0.3'] = (df_SPOT['None'] - df_SPOT['multiplicative_0.3_'+str(ii_b)]) #/ df_SPOT['None']

print (df_S2)

#fig, axs = plt.subplots(3,1, figsize=(32,10), sharey=True, sharex=True)

fig = plt.figure(figsize=(24,10))
gs = GridSpec(3,9, figure=fig)

axs = {}
axs['S2'] = {'bdo':fig.add_subplot(gs[0,0:7]),
                'additive':fig.add_subplot(gs[1,0:7]),
                'multiplicative':fig.add_subplot(gs[2,0:7])}
axs['SPOT'] = {'bdo':fig.add_subplot(gs[0,7:]),
                'additive':fig.add_subplot(gs[1,7:]),
                'multiplicative':fig.add_subplot(gs[2,7:])}
# do band dropout
data = []
for ii_b, band in enumerate(S2_bands):
    data.append(df_S2.loc[idx[:,band['name']],'band_dropout'].values.clip(-1,1))

bplot0S2 = axs['S2']['bdo'].boxplot(data, whis='range',patch_artist=True, medianprops = dict(linestyle='-', linewidth=1, color='firebrick'))
axs['S2']['bdo'].set_ylabel('IoU Impairment')
axs['S2']['bdo'].set_ylim([-1, 1.])
axs['S2']['bdo'].set_title('Band Dropout - Sentinel-2')
axs['S2']['bdo'].set_xticklabels([])
axs['S2']['bdo'].yaxis.set_major_formatter(mtick.PercentFormatter(1))

data = []
for ii_b, band in enumerate(SPOT_bands):
    data.append(df_SPOT[band['name']+'_bdo'].values.clip(-1,1))

bplot0SPOT = axs['SPOT']['bdo'].boxplot(data, whis='range',patch_artist=True, medianprops = dict(linestyle='-', linewidth=1, color='firebrick'))
#axs['SPOT']['bdo'].set_ylabel('IoU Impairment')
axs['SPOT']['bdo'].set_title('SPOT6/7')
axs['SPOT']['bdo'].set_ylim([-1, 1.])
axs['SPOT']['bdo'].set_xticklabels([])
axs['SPOT']['bdo'].yaxis.set_major_formatter(mtick.PercentFormatter(1))
axs['SPOT']['bdo'].set_yticklabels([])

# do additive 1,2,3
data = []
positions=[]
for ii_b, band in enumerate(S2_bands):
    data.append(df_S2.loc[idx[:,band['name']],'additive_0.1'].values.clip(-1,1))
    data.append(df_S2.loc[idx[:,band['name']],'additive_0.2'].values.clip(-1,1))
    data.append(df_S2.loc[idx[:,band['name']],'additive_0.3'].values.clip(-1,1))
    positions += [(ii_b+1 + (ii_p-1)/4) for ii_p in range(3)]

bplot1S2 = axs['S2']['additive'].boxplot(data, positions=positions, whis='range', widths=0.15, patch_artist=True, medianprops = dict(linestyle='-', linewidth=1, color='firebrick'))
axs['S2']['additive'].set_ylabel('IoU Impairment')
axs['S2']['additive'].set_ylim([-1, 1.])
axs['S2']['additive'].set_xticks(range(1,15))
axs['S2']['additive'].set_title('Additive Noise [10%, 20%, 30%] - Sentinel-2')
axs['S2']['additive'].set_xticklabels([])
axs['S2']['additive'].yaxis.set_major_formatter(mtick.PercentFormatter(1))

data = []
positions=[]
for ii_b, band in enumerate(SPOT_bands):
    data.append(df_SPOT[band['name']+'_additive_0.1'].values.clip(-1,1))
    data.append(df_SPOT[band['name']+'_additive_0.2'].values.clip(-1,1))
    data.append(df_SPOT[band['name']+'_additive_0.3'].values.clip(-1,1))
    positions += [(ii_b+1 + (ii_p-1)/4) for ii_p in range(3)]

bplot1SPOT = axs['SPOT']['additive'].boxplot(data, positions=positions, whis='range', widths=0.15, patch_artist=True, medianprops = dict(linestyle='-', linewidth=1, color='firebrick'))
axs['SPOT']['additive'].set_title('SPOT6/7')
axs['SPOT']['additive'].set_ylim([-1, 1.])
axs['SPOT']['additive'].set_xticks(range(1,5))
axs['SPOT']['additive'].set_xticklabels([])
axs['SPOT']['additive'].set_yticklabels([])
axs['SPOT']['additive'].yaxis.set_major_formatter(mtick.PercentFormatter(1))
axs['SPOT']['additive'].set_yticklabels([])


# do multiplicative 1,2,3
data = []
positions=[]
for ii_b, band in enumerate(S2_bands):
    data.append(df_S2.loc[idx[:,band['name']],'multiplicative_0.1'].values.clip(-1,1))
    data.append(df_S2.loc[idx[:,band['name']],'multiplicative_0.2'].values.clip(-1,1))
    data.append(df_S2.loc[idx[:,band['name']],'multiplicative_0.3'].values.clip(-1,1))
    positions += [(ii_b+1 + (ii_p-1)/4) for ii_p in range(3)]

bplot2S2 = axs['S2']['multiplicative'].boxplot(data, positions=positions, whis='range', widths=0.15,patch_artist=True, medianprops = dict(linestyle='-', linewidth=1, color='firebrick'))
axs['S2']['multiplicative'].set_xticks(range(1,15))
axs['S2']['multiplicative'].set_xticklabels(['{}$_{{'.format(band['name'])+'{}}}$'.format(band['resolution']) for band in S2_bands] )
axs['S2']['multiplicative'].set_ylabel('IoU Impairment')
axs['S2']['multiplicative'].set_ylim([-1, 1.])
axs['S2']['multiplicative'].set_title('Multiplicative Noise [10%, 20%, 30%] - Sentinel-2')
axs['S2']['multiplicative'].yaxis.set_major_formatter(mtick.PercentFormatter(1.))


data = []
positions=[]

for ii_b, band in enumerate(SPOT_bands):
    data.append(df_SPOT[band['name']+'_multiplicative_0.1'].values.clip(-1,1))
    data.append(df_SPOT[band['name']+'_multiplicative_0.2'].values.clip(-1,1))
    data.append(df_SPOT[band['name']+'_multiplicative_0.3'].values.clip(-1,1))
    positions += [(ii_b+1 + (ii_p-1)/4) for ii_p in range(3)]

bplot2SPOT = axs['SPOT']['multiplicative'].boxplot(data, positions=positions, whis='range', widths=0.15,patch_artist=True, medianprops = dict(linestyle='-', linewidth=1, color='firebrick'))
axs['SPOT']['multiplicative'].set_xticks(range(1,5))
axs['SPOT']['multiplicative'].set_xticklabels(['{}$_{{'.format(band['name'])+'{}}}$'.format(band['resolution']) for band in SPOT_bands])
axs['SPOT']['multiplicative'].set_title('SPOT6/7')
axs['SPOT']['multiplicative'].set_ylim([-1, 1.])
axs['SPOT']['multiplicative'].yaxis.set_major_formatter(mtick.PercentFormatter(1.))
axs['SPOT']['multiplicative'].set_yticklabels([])


#for bplot in (bplot1, bplot2):
for patch, band in zip(bplot0S2['boxes'], S2_bands):
    patch.set_facecolor(gg_colors[band['color']])

for patch, band in zip(bplot0SPOT['boxes'], SPOT_bands):
    patch.set_facecolor(gg_colors[band['color']])

for patch, band in zip(bplot1S2['boxes'], list(np.repeat(S2_bands,3))):
    patch.set_facecolor(gg_colors[band['color']])

for patch, band in zip(bplot1SPOT['boxes'], list(np.repeat(SPOT_bands,3))):
    patch.set_facecolor(gg_colors[band['color']])

for patch, band in zip(bplot2S2['boxes'], list(np.repeat(S2_bands,3))):
    patch.set_facecolor(gg_colors[band['color']])

for patch, band in zip(bplot2SPOT['boxes'], list(np.repeat(SPOT_bands,3))):
    patch.set_facecolor(gg_colors[band['color']])

fig.savefig('./band_perturbation.png')
plt.show()