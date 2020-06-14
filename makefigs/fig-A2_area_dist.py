import geopandas as gpd
from area import area
from shapely import geometry
gpd.options.use_pygeos=False
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

root = os.getcwd()

trn_polys = gpd.read_file(os.path.join(root,'data','all_trn_polygons.geojson'))
cv_polys = gpd.read_file(os.path.join(root,'data','cv_all_polys.geojson'))
ne = gpd.read_file(os.path.join(root,'data','ne_10m_countries.gpkg'))
cv_tiles = gpd.read_file(os.path.join(root,'data','data/cv_all_tiles.geojson'))

gdf = gpd.read_file('./data/ABCD_finalized.geojson')

trn_polys['area'] = trn_polys.apply(lambda row: area(geometry.mapping(row['geometry'])), axis=1)

cv_polys['area'] = cv_polys.apply(lambda row: area(geometry.mapping(row['geometry'])), axis=1)


fig, ax = plt.subplots(1,1,figsize=(6,3))
np.log10(trn_polys['area']).hist(ax=ax, bins=10, alpha=0.75, edgecolor='k',histtype='step', linewidth=2,density=True, fill=False)
np.log10(cv_polys['area']).hist(ax=ax, bins=10, alpha=0.75, edgecolor='k',histtype='step', linewidth=2,density=True, linestyle='--',fill=False)
np.log10(gdf['area']).hist(ax=ax, bins=10, alpha=0.75, edgecolor='r',histtype='step', linewidth=2,density=True, linestyle='-',fill=False)

ax.grid(False)
ax.set_yticks([])
ax.set_xlabel('Installation Area $[m^2]$')
ax.set_ylabel('Freq')
ax.set_xlim([0, 7])
ax.set_xticks([ii for ii in range(1,8)])
ax.set_xticklabels(['10$^{}$'.format(ii) for ii in range(1,8)])


custom_lines = [
    Line2D([0],[0],color='k',marker=None,linestyle='-', lw=2),
    Line2D([0],[0],color='k',marker=None,linestyle='--', lw=2),
    Line2D([0],[0],color='r',marker=None,linestyle='-', lw=2),
]

#axs['legend'].axis('off')
lgd = ax.legend(custom_lines, ['Training Set','Cross-Validation Set','Inferred Set'], ncol=3, loc='center',bbox_to_anchor=(0.5,-0.3), frameon=False)
plt.savefig(os.path.join(root,'makefigs','figures','fig-A2_trn_cv_area.png'), bbox_extra_artists=(lgd,))

plt.show()