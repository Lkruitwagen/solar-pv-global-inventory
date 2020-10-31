import geopandas as gpd
from area import area
from shapely import geometry
gpd.options.use_pygeos=False
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import os

root = os.getcwd()

trn_polys = gpd.read_file(os.path.join(root,'data','all_trn_polygons.geojson'))
cv_polys = gpd.read_file(os.path.join(root,'data','cv_all_polys.geojson'))
ne = gpd.read_file(os.path.join(root,'data','ne_10m_countries.gpkg'))
cv_tiles = gpd.read_file(os.path.join(root,'data','cv_all_tiles.geojson'))

gdf = gpd.read_file(os.path.join(root,'data','ABCD_finalized.geojson'))

trn_polys['area'] = trn_polys.apply(lambda row: area(geometry.mapping(row['geometry'])), axis=1)

cv_polys['area'] = cv_polys.apply(lambda row: area(geometry.mapping(row['geometry'])), axis=1)


#fig, ax = plt.subplots(1,1,figsize=(4,2))

fig = plt.figure(figsize=(4,2))

gs = GridSpec(5, 1, figure=fig, wspace=0.1, hspace=0.1)
axs = {}
axs['plot']=fig.add_subplot(gs[0:4,0])
axs['legend'] = fig.add_subplot(gs[4,0])
axs['legend'].axis('off')

np.log10(trn_polys['area']).hist(ax=axs['plot'], bins=10, alpha=0.75, edgecolor='k',histtype='step', linewidth=2,density=True, fill=False)
np.log10(cv_polys['area']).hist(ax=axs['plot'], bins=10, alpha=0.75, edgecolor='k',histtype='step', linewidth=2,density=True, linestyle='--',fill=False)
np.log10(gdf['area']).hist(ax=axs['plot'], bins=10, alpha=0.75, edgecolor='r',histtype='step', linewidth=2,density=True, linestyle='-',fill=False)

axs['plot'].grid(False)
axs['plot'].set_yticks([])
axs['plot'].set_xlabel('Installation Area $[m^2]$')
axs['plot'].set_ylabel('Freq')
axs['plot'].set_xlim([0, 7])
axs['plot'].set_xticks([ii for ii in range(1,8)])
axs['plot'].set_xticklabels(['10$^{}$'.format(ii) for ii in range(1,8)])


custom_lines = [
    Line2D([0],[0],color='k',marker=None,linestyle='-', lw=2),
    Line2D([0],[0],color='k',marker=None,linestyle='--', lw=2),
    Line2D([0],[0],color='r',marker=None,linestyle='-', lw=2),
]

#axs['legend'].axis('off')
lgd = axs['legend'].legend(custom_lines, ['Training Set','Cross-Validation Set','Final Predicted Set'], ncol=3, bbox_to_anchor=(0.5,0.0),loc='center',frameon=False)

#plt.savefig(os.path.join(root,'makefigs','figures','fig-A2_trn_cv_area.png'))#, bbox_extra_artists=(lgd,))
plt.show()
