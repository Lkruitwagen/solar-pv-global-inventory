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


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

        # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap



root = os.getcwd()

ne = gpd.read_file(os.path.join(root,'data','ne_10m_countries.gpkg'))
ne = ne[~ne['geometry'].isna()]
ne = ne.set_index('ISO_A2', drop=False)

gdf = gpd.read_file(os.path.join(root,'data','SPV_newmw.gpkg'))
df = pd.read_csv(os.path.join(root,'data','tabula-irena.csv')).set_index('iso2')
df['2018'] = df['2018'].str.replace(' ','').astype(float)

bins = {
    0:{'min':10**2,'max':10**3,'recall':.18},
    1:{'min':10**3,'max':10**4,'recall':.25},
    2:{'min':10**4,'max':10**5,'recall':.71},
    3:{'min':10**5,'max':10**6,'recall':.82},
    4:{'min':10**6,'max':10**10,'recall':.88},
}


gdf['adj_mw'] = np.nan

for kk,vv in bins.items():
    gdf.loc[(gdf['area']>=vv['min'])&(gdf['area']<vv['max']),'adj_mw'] = gdf.loc[(gdf['area']>=vv['min'])&(gdf['area']<vv['max']),'capacity_mw'] /vv['recall']


#fig, axs = plt.subplots(2,1,figsize=(18,15))

fig = plt.figure(figsize=(18,15))

gs = GridSpec(17, 1, figure=fig, wspace=0.3, hspace=0.2)

axs = {}

axs['base'] = fig.add_subplot(gs[0:8,:])
axs['adj'] = fig.add_subplot(gs[8:16,:])
axs['leg'] = fig.add_subplot(gs[16,:])


#plot basemap
ne.plot(ax=axs['base'], color='#d1d1d1')
ne.plot(ax=axs['adj'], color='#d1d1d1')

cmap = mpl.colors.ListedColormap(['#00ffff','#ff00ff','#ffff00'])
shifted_cmap = shiftedColorMap(cmap, midpoint=0.75, name='shifted')
print ('cm',shifted_cmap(0))
print ('cm',shifted_cmap(0.5))
print ('cm',shifted_cmap(0.75))
print ('cm',shifted_cmap(1))
print ('cm',shifted_cmap(0.691704))
print ('cm',shifted_cmap(0.291704))

ne = ne.merge(gdf[['capacity_mw','adj_mw','iso-3166-1']].groupby('iso-3166-1').sum(), left_index=True, right_index=True)
#print (ne)
ne = ne.merge(df[['2018']], left_index=True, right_index=True)
#print (ne)

ne['yield'] = ne['capacity_mw']/ne['2018']
ne['adj_yield'] = ne['adj_mw']/ne['2018']

#print (ne)
VMIN=0.5
VMAX=1.25

def map_color(val):
  if val>=1:
    return cm.spring(np.clip((val-1)/(VMAX-1),0,1))
  else:
    return cm.cool(np.clip((val-VMIN)/(1-VMIN),0,1))

pd.DataFrame(ne[['capacity_mw','2018','yield']]).to_csv(os.path.join(root,'data','meow.csv'))



ne['color'] = ne['yield'].apply(lambda el: map_color(el))
ne['adj_color'] = ne['adj_yield'].apply(lambda el: map_color(el))
print (ne)

ne['x'] = ne['geometry'].apply(lambda el: el.representative_point().x)
ne['y'] = ne['geometry'].apply(lambda el: el.representative_point().y)

print (np.log10(ne['capacity_mw']*1000))

delta = 0.01

#levels = np.linspace(VMIN, VMAX, int((VMAX-VMIN)*10))
levels = [VMIN+delta*ii-delta/2 for ii in range(int((VMAX-VMIN)/delta)+2)]
print ('levels',levels)

cols = [map_color(val) for val in levels]

cmaplevels, norm = from_levels_and_colors(levels,cols[:-1])


axs['base'].scatter(ne['x'], ne['y'], s = ne['capacity_mw']/50, c = ne['color'], edgecolor='#636363', linewidth=2)
axs['adj'].scatter(ne['x'], ne['y'], s = ne['adj_mw']/50, c = ne['adj_color'], edgecolor='#636363', linewidth=2)

axs['adj'].scatter([-170,-170], [-30,-45], s = [10000/50,100000/50], c = '#ababab', edgecolor='#636363', linewidth=2)
axs['adj'].text(-160,-30,'10GW', va='center')
axs['adj'].text(-160,-45,'100GW', va='center')

#norm = cm.colors.Normalize(vmax=VMAX, vmin=VMIN)

xticks = [VMIN+0.05*ii for ii in range(int((VMAX-VMIN)/0.05)+1)]
print (xticks)
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmaplevels), cax=axs['leg'],ticks = xticks,  orientation='horizontal')
cbar.ax.set_xticklabels([f'{tick*100:.0f}%' for tick in xticks])
cbar.ax.set_xlabel('Aggregate capacity as a portion of capacity reported by IRENA')

#axs['leg'].set_xticks(xticks)

#plot_polygon_collection(axs[0], ne.loc[:,'geometry'], color=ne.loc[:,'color'])
#plot_polygon_collection(axs[1], ne.loc[:,'geometry'], color=ne.loc[:,'adj_color'])

#ne.plot(ax=axs[0], column='yield', cmap=shifted_cmap,vmin=0.5, vmax=1.5)




"""
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
"""


axs['base'].set_ylim([-60,85])
axs['base'].set_xticks([])
axs['base'].set_yticks([])
axs['adj'].set_ylim([-60,85])
axs['adj'].set_xticks([])
axs['adj'].set_yticks([])
axs['base'].set_title('(a) Aggregate capacity by country')
axs['adj'].set_title('(b) Aggregate capacity adjusted for area-binned recall')
plt.savefig(os.path.join(root,'makefigs','figures','fig-A12_deploy_agg_recall.png'))
plt.show()