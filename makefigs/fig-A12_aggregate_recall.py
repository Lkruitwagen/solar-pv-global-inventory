import geopandas as gpd
gpd.options.use_pygeos=False
import pandas as pd
import os, json, geojson
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
from geopandas.plotting import plot_polygon_collection
from matplotlib.gridspec import GridSpec
from matplotlib.colors import from_levels_and_colors


root = os.getcwd()

#ne = gpd.read_file(os.path.join(root,'data','ne_10m_countries.gpkg'))
#ne = ne[~ne['geometry'].isna()]
#ne = ne.set_index('ISO_A2', drop=False)

bins = {
    0:{'min':10**2,'max':10**3,'recall_T':66, 'recall_F':1645, 'precision_T':87,'precision_F':3, 'area_mu':0.37701015880980093, 'area_std': 0.6370519408706289},
    1:{'min':10**3,'max':10**4,'recall_T':1091, 'recall_F':1465, 'precision_T':934,'precision_F':39, 'area_mu':0.1994739505469657,'area_std': 0.7217854444253359},
    2:{'min':10**4,'max':10**5,'recall_T':1819, 'recall_F':241, 'precision_T':1305,'precision_F':14, 'area_mu':0.10893815502345362,'area_std': 0.5129571834277274},
    3:{'min':10**5,'max':10**6,'recall_T':553, 'recall_F':20, 'precision_T':521,'precision_F':10, 'area_mu':-0.012558514210915256, 'area_std': 0.28063800305152226},
    4:{'min':10**6,'max':10**10,'recall_T':62, 'recall_F':4, 'precision_T':70,'precision_F':2, 'area_mu':-0.024982255186530276, 'area_std': 0.20184961361379802},
}

ne = gpd.read_file(os.path.join(root,'data','ne_10m_countries.gpkg'))
ne = ne[~ne['geometry'].isna()]
ne = ne.set_index('ISO_A2', drop=False)

gdf = gpd.read_file(os.path.join(root,'data','SPV_v5.gpkg'))
df = pd.read_csv(os.path.join(root,'data','tabula-irena.csv')).set_index('iso2')
df['2018'] = df['2018'].str.replace(' ','').astype(float)

gdf['adj_mw'] = np.nan

for kk,vv in bins.items():
    vv['recall'] = vv['recall_T']/(vv['recall_T']+vv['recall_F'])
    vv['precision'] = vv['precision_T']/(vv['precision_T']+vv['precision_F'])
    gdf.loc[(gdf['area']>=vv['min'])&(gdf['area']<vv['max']),'adj_mw'] = gdf.loc[(gdf['area']>=vv['min'])&(gdf['area']<vv['max']),'capacity_mw'] * vv['precision'] /vv['recall']


#fig, axs = plt.subplots(2,1,figsize=(18,15))

fig = plt.figure(figsize=(12,10))

gs = GridSpec(16, 1, figure=fig, wspace=0.3, hspace=0.2)

axs = {}

axs['base'] = fig.add_subplot(gs[0:16,:])
#axs['adj'] = fig.add_subplot(gs[9:17,:])
for kk in ['base']:
    axs[kk].set_xscale('log')
    axs[kk].set_yscale('log')
#axs['leg'] = fig.add_subplot(gs[16,:])

offsets = {
    'MX':{'x':-850,'y':-120},
    'SK':{'x':-100,'y':250},
    'PA':{'x':-40,'y':0},
    'BY':{'x':-30,'y':140},
    'PR':{'x':0,'y':100},
    'SV':{'x':10,'y':45},
    'MR':{'x':-25,'y':20},
    'CO':{'x':-15,'y':-10},
    'MU':{'x':-10,'y':10},
    'MG':{'x':0,'y':-10},
    'EC':{'x':0,'y':-10},
}


ne = ne.merge(gdf[['capacity_mw','adj_mw','iso-3166-1']].groupby('iso-3166-1').sum(), left_index=True, right_index=True)
#print (ne)
ne = ne.merge(df[['2018']], left_index=True, right_index=True)
#print (ne)

ne = ne.sort_values('capacity_mw', ascending=False).iloc[0:90,:]
ne[['2018','capacity_mw']].rename(columns={'2018':'IRENA_2018','capacity_mw':'ours_2018'}).to_csv(os.path.join(os.getcwd(),'makefigs','data','fig-A12.csv'))

axs['base'].plot([0, 1], [0, 1], transform=axs['base'].transAxes, ls='--', c='gray',zorder=-1)
#axs['adj'].plot([0, 1], [0, 1], transform=axs['adj'].transAxes, ls='--', c='gray')
axs['base'].scatter(ne.iloc[0:90,ne.columns.get_loc('2018')], ne.iloc[0:90,ne.columns.get_loc('capacity_mw')], s=150, color='w', edgecolors='k', zorder=1)
#axs['adj'].scatter(ne.iloc[0:90,ne.columns.get_loc('2018')], ne.iloc[0:90,ne.columns.get_loc('adj_mw')])

arrowprops=dict(arrowstyle="-", connectionstyle="arc3")

for idx, row in ne.iterrows():
    if idx in offsets.keys():
        axs['base'].annotate(idx,xy=(row['2018'], row['capacity_mw']), xycoords='data',xytext=(row['2018']+offsets[idx]['x'], row['capacity_mw']+offsets[idx]['y']), textcoords='data', arrowprops=arrowprops, fontsize=6)
    else:
        axs['base'].text(row['2018'],row['capacity_mw'],idx, ha='center', va='center', fontsize=6)

axs['base'].set_xlabel('IRENA Capacity 2018 [MW]')
axs['base'].set_ylabel('Predicted Set Capacity (ours) [MW]')
axs['base'].set_xlim([1,3e5])
axs['base'].set_ylim([1,3e5])

#axs['base'].set_title('Aggregate capacity by country vs IRENA 2018')
#axs['adj'].set_title('(b) Aggregate capacity adjusted for area-binned recall vs IRENA 2018')
plt.savefig(os.path.join(root,'makefigs','figures','fig-A12_deploy_agg_recall.png'))
plt.show()