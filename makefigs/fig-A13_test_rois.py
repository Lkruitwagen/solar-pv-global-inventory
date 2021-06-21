import geopandas as gpd
gpd.options.use_pygeos=False
import pandas as pd
import os, json, geojson
import glob
from shapely import geometry
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
from geopandas.plotting import plot_polygon_collection
from matplotlib.gridspec import GridSpec
from matplotlib.colors import from_levels_and_colors
from matplotlib.markers import MarkerStyle
from area import area


root = os.getcwd()

ne = gpd.read_file(os.path.join(root,'data','ne_10m_countries.gpkg'))
ne = ne[~ne['geometry'].isna()]
ne = ne.set_index('ISO_A2', drop=False)

test_tiles = gpd.read_file(os.path.join(root,'data','testset_aois.geojson'))
test_polys = gpd.read_file(os.path.join(root,'data','test_set_handlabelled.geojson'))
test_polys['geoarea'] = test_polys['geometry'].apply(lambda geom: area(geometry.mapping(geom)))

df = pd.DataFrame(test_polys.groupby('aoi').size().astype(str)).rename(columns={0:'N'})
df['N'] = df['N'].astype(int)
df['geoarea'] = test_polys[['aoi','geoarea']].groupby('aoi').sum()
df['mean_area'] = df['geoarea']/df['N'].astype(int)

test_tiles['area'] = test_tiles['geometry'].apply(lambda geom: area(geometry.mapping(geom)))
test_tiles.index=test_tiles['idx'].str.split('_').str[0]

df = pd.merge(test_tiles, df, how='left', left_index=True, right_index=True).rename(columns={0:'size'})

cm_size = cm.spring_r
cm_n = cm.summer_r
min_N, max_N = np.log10(1),np.log10(800)
min_A, max_A = np.log10(1e7), np.log10(5e10) # 10km^2, 50k km^2
min_mA, max_mA = np.log10(5e2), np.log10(5e6)
min_S, max_S = 5,100

df['log10_mean_area'] = np.log10(df['mean_area'])
df['log10_N'] = np.log10(df['N'])
df['log10_A'] = np.log10(df['area'])

df['s'] = min_S + (df['log10_A'] - min_A)/(max_A-min_A) * (max_S-min_S)
df['cint_N'] = (df['log10_N'] - min_N)/(max_N-min_N)
df['cint_mA'] = (df['log10_mean_area'] - min_mA)/(max_mA-min_mA)
df['s'] = df['s'].clip(min_S, max_S)
df['cint_N'] = df['cint_N'].clip(0,1)
df['cint_mA'] = df['cint_mA'].clip(0,1)
df['c_N'] = df['cint_N'].apply(lambda el: '#'+''.join([f'{int(255*val):02X}' for val in cm_n(el)[0:3]]))
df['c_mA'] = df['cint_mA'].apply(lambda el: '#'+''.join([f'{int(255*val):02X}' for val in cm_size(el)[0:3]]))

fig = plt.figure(figsize=(12,7))

gs = GridSpec(24, 5, figure=fig, wspace=0.3, hspace=0.2)

axs = {}

axs['base'] = fig.add_subplot(gs[0:19,:])
axs['cmap_N'] = fig.add_subplot(gs[19,1:])
axs['cmap_mA'] = fig.add_subplot(gs[22,1:])
axs['N'] = fig.add_subplot(gs[20:,0])

ne.plot(ax=axs['base'], color='#e6e6e6', edgecolor=None)

axs['base'].scatter(
    test_tiles.centroid.x, 
    test_tiles.centroid.y, 
    s=df['s'].values,
    marker=MarkerStyle('o', fillstyle='left'), 
    color=df['c_N'].values,
)
axs['base'].scatter(
    test_tiles.centroid.x, 
    test_tiles.centroid.y, 
    s=df['s'].values,
    marker=MarkerStyle('o', fillstyle='right'), 
    color=df['c_mA'].values,
)
axs['base'].set_ylim([-60,85])
axs['base'].set_xticks([])
axs['base'].set_yticks([])

norm_size = cm.colors.Normalize(vmax=max_mA, vmin=min_mA)
norm_N = cm.colors.Normalize(vmax=max_N, vmin=min_N)
cbar_size = fig.colorbar(cm.ScalarMappable(norm=norm_size, cmap=cm_size), orientation='horizontal',cax=axs['cmap_mA'])
cbar_N = fig.colorbar(cm.ScalarMappable(norm=norm_N, cmap=cm_n), orientation='horizontal',cax=axs['cmap_N'])

cbar_N.set_ticks([np.log10(ii) for ii in [1,10,100,500]]) 
cbar_N.set_ticklabels([f'{ii}' for ii in [1,10,100,500]]) 
axs['cmap_N'].set_title('Left: number of detections')
cbar_size.set_ticks([np.log10(ii) for ii in [5e2, 5e3, 5e4, 5e5, 5e6]]) #m^2
cbar_size.set_ticklabels([f'{ii}' for ii in ['500', '5k', '50k', '500k', '5mn']]) #m^2
axs['cmap_mA'].set_title('Right: mean sample size [m$^2$]')

axs['N'].set_title('RoI Size [km$^2$]')
axs['N'].scatter([0.5,0.5],[10,0],s=[min_S,max_S],c='#cccccc')
axs['N'].text(x=0.505,y=10,s='10', ha='left',va='center')
axs['N'].text(x=0.505,y=0,s='50k', ha='left',va='center')
axs['N'].set_ylim([-5,15])
axs['N'].axis('off')

plt.savefig(os.path.join(root,'makefigs','figures','fig-A13_test_rois.png'))
plt.show()