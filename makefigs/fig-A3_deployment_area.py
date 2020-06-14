import geopandas as gpd
from area import area
from shapely import geometry
gpd.options.use_pygeos=False
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


root = os.getcwd()

ne = gpd.read_file(os.path.join(root,'data','ne_10m_countries.gpkg'))
popshp = gpd.read_file(os.path.join(root,'data','popshp_gt1_d7k.geojson'))
dnr = gpd.read_file(os.path.join(root,'data','do_not_run.geojson'))

dnr_mp = dnr.unary_union
popshp = popshp[~popshp.geometry.isna()]

# clip Russia
RU = ne.loc[ne['ISO_A2']=='RU','geometry']
RU_clip = geometry.Polygon([[-180,60],[-180,89],[180,89],[180,60]])
RU_elim = RU.geometry.intersection(RU_clip)
RU_elim = RU_elim.geometry.unary_union

#clip North America
NA_poly = geometry.Polygon([[-169,60],[-169,89],[-30,89],[-30,60]])

# clip all geometries
popshp.geometry = popshp.geometry.apply(lambda geom: geom.difference(dnr_mp))
popshp.geometry = popshp.geometry.apply(lambda geom: geom.difference(NA_poly))
popshp.geometry = popshp.geometry.apply(lambda geom: geom.difference(RU_elim))
popshp[~popshp.geometry.is_empty].plot()

# plot
fig, ax = plt.subplots(1,1,figsize=(30,24))
ne.plot(ax=ax, color='#cccccc')
popshp[~popshp.geometry.is_empty].plot(ax=ax, color='g')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim([-60,85])
ax.set_xlim([-169,180])
legend_elements = [Patch(facecolor='g',  label='Pipeline Deployment Area')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=24, frameon=False)
plt.savefig(os.path.join(root,'makefigs','figures','fig-A3_deployment_area.png')
plt.show()