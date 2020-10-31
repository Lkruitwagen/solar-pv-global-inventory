import geopandas as gpd
gpd.options.use_pygeos=False
import pandas as pd
import os, json, geojson
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from geopandas.plotting import plot_polygon_collection


root = os.getcwd()

ne = gpd.read_file(os.path.join(root,'data','ne_10m_countries.gpkg'))
ne_prov = gpd.read_file(os.path.join(root,'data','ne_10m_admin_1_states_provinces.geojson'))

#ne['N_obs_SPOT'] = np.nan
#ne['N_T_SPOT'] = np.nan
#ne_prov['N_obs_SPOT'] = np.nan
#ne_prov['N_T_SPOT'] = np.nan

ne_prov = ne_prov.set_index('iso_3166_2', drop=False)
ne = ne.set_index('ISO_A2', drop=False)

do_prov = ['AU','BR','CA','CN','IN','US']

label_df = pd.read_csv(os.path.join(root,'data','label_df.csv'))

SPOT_country_df = pd.read_csv(os.path.join(root,'data','ne_SPOT.csv')).set_index('ISO_A2')
SPOT_prov_df = pd.read_csv(os.path.join(root,'data','ne_prov_SPOT.csv')).set_index('iso_3166_2')

# do SPOT merging
#ne[['ISO_A2','N_obs_SPOT','N_T_SPOT']]
#ne_prov[['iso_3166_2','N_obs_SPOT','N_T_SPOT']]

#print (ne[['ISO_A2','N_obs_SPOT','N_T_SPOT']])
#print (ne_prov[['iso_3166_2','N_obs_SPOT','N_T_SPOT']])

ne = ne.merge(SPOT_country_df, how='left',left_index=True,right_index=True)
print ('ne',ne)
ne_prov = ne_prov.merge(SPOT_prov_df, how='left',left_index=True,right_index=True)
print ('ne_prov',ne_prov)
ne['N_T_SPOT'] = ne['N_T_SPOT'].fillna(0)

# do S2 merging
ne = ne.merge(pd.DataFrame(label_df.groupby('iso2').size()), how='left',left_index=True,right_index=True).rename(columns={0:'N_obs_S2'})
ne = ne.merge(pd.DataFrame(label_df[['iso2','label']].groupby('iso2').sum()), how='left',left_index=True,right_index=True).rename(columns={'label':'N_T_S2'})
ne_prov = ne_prov.merge(pd.DataFrame(label_df.groupby('iso_prov').size()), how='left',left_index=True,right_index=True).rename(columns={0:'N_obs_S2'})
ne_prov = ne_prov.merge(pd.DataFrame(label_df[['iso_prov','label']].groupby('iso_prov').sum()), how='left',left_index=True,right_index=True).rename(columns={'label':'N_T_S2'})

ne['por_S2'] = ne['N_T_S2']/ne['N_obs_S2']
ne['por_SPOT'] = ne['N_T_SPOT']/ne['N_obs_SPOT']
ne_prov['por_S2'] = ne_prov['N_T_S2']/ne_prov['N_obs_S2']
ne_prov['por_SPOT'] = ne_prov['N_T_SPOT']/ne_prov['N_obs_SPOT']

ne['por_SPOT'] = ne['por_SPOT'].fillna(0)
ne_prov['por_SPOT'] = ne_prov['por_SPOT'].fillna(0)

ne['log10_obs_S2'] = np.log10(ne['N_obs_S2'])
ne['log10_obs_SPOT'] = np.log10(ne['N_obs_SPOT'])
ne_prov['log10_obs_S2'] = np.log10(ne_prov['N_obs_S2'])
ne_prov['log10_obs_SPOT'] = np.log10(ne_prov['N_obs_SPOT'])
ne['log10_obs_SPOT'] = ne['log10_obs_SPOT'].fillna(0)

vmax_S2=4
vmax_SPOT=4.5
por_max_S2 = 1
por_max_SPOT = 0.5

def conv_rgb_S2(row):
    if np.isnan(row['log10_obs_S2']):
        return [0,0,0,1] #'#%02x%02x%02x' % 
    else:
        b = row['log10_obs_S2']/vmax_S2
        r = row['por_S2']*b
        g = (1-row['por_S2'])*b
    return [r,g,b,1] #'#%02x%02x%02x' % 

def conv_rgb_SPOT(row):
    if np.isnan(row['log10_obs_SPOT']):
        return [0,0,0,1] #'#%02x%02x%02x' % 
    else:
        b = row['log10_obs_SPOT']/vmax_SPOT
        r = np.clip(row['por_SPOT']/por_max_SPOT,0,1)*b
        g = (1-np.clip(row['por_SPOT']/por_max_SPOT,0,1))*b
        
    arr = np.array([r,g,b,1])
    if ((arr>1).sum()+ (arr<0).sum())>0:
        print (row)
    return [r,g,b,1] #'#%02x%02x%02x' % 

ne['color_S2'] = ne.apply(lambda row: conv_rgb_S2(row), axis=1)
ne_prov['color_S2'] = ne_prov.apply(lambda row: conv_rgb_S2(row), axis=1)
ne['color_SPOT'] = ne.apply(lambda row: conv_rgb_SPOT(row), axis=1)
ne_prov['color_SPOT'] = ne_prov.apply(lambda row: conv_rgb_SPOT(row), axis=1)

ne = ne[~ne.geometry.isna()]

def leg_gen(dim):
    a = np.stack([np.linspace(0,1,dim)]*dim)
    P = np.linspace(1,0,dim)
    R = np.stack([P]*dim).T
    G = 1-R
    B = np.ones((dim,dim))
    return np.moveaxis(np.stack([R,G,B]),0,-1)*np.moveaxis(np.stack([a]*3),0,-1)

leg_arr = leg_gen(50)


fig, axs = plt.subplots(2,1,figsize=(18,15))

#plot basemap
ne.plot(ax=axs[0], color='#d1d1d1')
ne.plot(ax=axs[1], color='#d1d1d1')

#plot polys S2
plot_polygon_collection(axs[0], ne.loc[~ne['ISO_A2'].isin(do_prov),'geometry'], color=ne.loc[~ne['ISO_A2'].isin(do_prov),'color_S2'])
plot_polygon_collection(axs[0], ne_prov.loc[ne_prov['iso_a2'].isin(do_prov),'geometry'], color=ne_prov.loc[ne_prov['iso_a2'].isin(do_prov),'color_S2'])

#plopt polys SPOT
plot_polygon_collection(axs[1], ne.loc[~ne['ISO_A2'].isin(do_prov),'geometry'], color=ne.loc[~ne['ISO_A2'].isin(do_prov),'color_SPOT'])
plot_polygon_collection(axs[1], ne_prov.loc[ne_prov['iso_a2'].isin(do_prov),'geometry'], color=ne_prov.loc[ne_prov['iso_a2'].isin(do_prov),'color_SPOT'])

#ne.loc[(~ne['por'].isna()) &(ne['N_T']>5 ) & (~ne['ISO_A2'].isin(do_prov)),:].plot(ax=axs[0], column='por', cmap='magma')
#ne_prov.loc[(~ne_prov['por'].isna()) & (ne_prov['N_T']>5) & (ne_prov['iso_a2'].isin(do_prov)),:].plot(ax=axs[0], column='por', cmap='magma')
#ne.loc[(~ne['por'].isna()) &(ne['N_T']<=5 ) & (~ne['ISO_A2'].isin(do_prov)),:].plot(ax=axs[0], column='log10_obs', cmap='bone', vmax=4)
#ne_prov.loc[(~ne_prov['por'].isna()) & (ne_prov['N_T']<=5) & (ne_prov['iso_a2'].isin(do_prov)),:].plot(ax=axs[0], column='log10_obs', cmap='bone', vmax=4)
ins1 = axs[0].inset_axes([0,0.12,0.2,0.25])
ins2 = axs[1].inset_axes([0,0.12,0.2,0.25])

ins1.imshow(leg_arr)
ins2.imshow(leg_arr)

ins1.set_xticks([ii*(50/4.) for ii in range(5)])
ins1.set_xticklabels([f'10$^{ii}$' for ii in range(5)])
ins1.set_yticks([ii*12.5 for ii in range(5)])
ins1.set_yticklabels([str(1-ii*12.5/50) for ii in range(5)])
ins1.set_ylabel('Precision')
ins1.set_xlabel('N$_{Observations}$')

ins2.set_xticks([ii*(45/4) for ii in range(5)])
ins2.set_xticklabels([f'10$^{ii}$' for ii in range(5)])
ins2.set_yticks([ii*(50/4) for ii in range(5)])
ins2.set_yticklabels(['0.0','0.125','0.25','0.375','>0.5'][::-1])
ins2.set_ylabel('Precision')
ins2.set_xlabel('N$_{Observations}$')



axs[0].set_ylim([-60,85])
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[1].set_ylim([-60,85])
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[0].set_title('(a) Sentinel-2')
axs[1].set_title('(b) SPOT6/7')
plt.savefig(os.path.join(root,'makefigs','figures','fig-A7_deploy_precision.png'))
plt.show()