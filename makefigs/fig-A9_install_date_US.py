import os, sys, glob, pickle
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
import numpy as np

root=os.getcwd()

match_files = glob.glob(os.path.join(root,'data','matches','match_wri_*_4000_0.05*.gpkg'))
match_files += glob.glob(os.path.join(root,'data','matches','match_eia*.gpkg'))
match_files = [f for f in match_files if 'wri_US' not in f]

match_df = gpd.GeoDataFrame(pd.concat([gpd.read_file(f) for f in match_files]))


gdf = gpd.read_file(os.path.join(root,'data','ABCD_landcover.geojson')) # for EIA matching

gdf = gdf.reset_index().rename(columns={'index':'unique_id'})

gdf = gdf.merge(match_df[['unique_id','match_id']], how='left',left_on='unique_id',right_on='unique_id')

gdf['install_date'] = gdf['install_date'].str.replace('<2016-06','2000-01-01')
gdf['install_date'] = gdf['install_date'].str.replace(',','')
gdf['install_date'] = gdf['install_date'].str[0:10]

gdf['dt_obj'] = pd.to_datetime(gdf['install_date'])


wri_df = pd.read_csv(os.path.join(root, 'data','WRI_gppd.csv'))
eia_df = gpd.read_file(os.path.join(root,'data','eia_powerstations','PowerPlants_US_202001.shp'))
eia_meta = pd.read_csv(os.path.join(root,'data','eia_powerstations','february_generator2020.csv'))
iso2 = pd.read_csv(os.path.join(root,'data','iso2.csv'))
wri_df = wri_df.merge(iso2[['iso3','iso2']], how='left',left_on='country',right_on='iso3')

eia_meta.dropna(subset=['Plant ID'], inplace=True)
eia_meta['Plant ID'] = eia_meta['Plant ID'].astype(int)
eia_meta['Nameplate Capacity (MW)'] = eia_meta['Nameplate Capacity (MW)'].str.replace(',','').astype(float)
eia_meta['dt_string'] = eia_meta['Operating Year'].map('{:.0f}'.format) +'-'+ eia_meta['Operating Month'].astype(int).map('{:02d}'.format)+'-01'
eia_meta['dt_obj'] = pd.to_datetime(eia_meta['dt_string'])
eia_meta['dt_ns']= eia_meta['dt_obj'].astype(int)
eia_groupplant = eia_meta[['Plant ID', 'Nameplate Capacity (MW)']].groupby('Plant ID').sum()

def avg_datetime(series):
    dt_min = series.min()
    deltas = [x-dt_min for x in series]
    return dt_min + functools.reduce(operator.add, deltas) / len(deltas)

eia_groupplant['dt_ns'] = eia_meta[['Plant ID', 'dt_ns']].groupby('Plant ID').mean()
eia_groupplant['dt_ns'] = pd.to_datetime(eia_groupplant['dt_ns'])

eia_df = eia_df.merge(eia_groupplant, how='left',left_on='Plant_Code',right_index=True)

dated_slice = gdf.loc[(gdf['iso-3166-1']=='US')&(gdf['dt_obj']>dt.strptime('2016-06-01','%Y-%m-%d'))&(gdf['dt_obj']<dt.strptime('2018-12-31','%Y-%m-%d'))&(gdf['match_id']!=''),['dt_obj','match_id']]


eia_df['Plant_Code'] = eia_df['Plant_Code'].astype(str)
eia_df['plant_match'] = 'EIA'+eia_df['Plant_Code'].astype(str)

dated_slice = dated_slice.merge(eia_df[['plant_match','dt_ns']], how='left',left_on='match_id',right_on='plant_match')

inds  = (dated_slice['dt_ns']>dt.strptime('2016-06-01','%Y-%m-%d')) & (dated_slice['dt_ns']<dt.strptime('2018-12-31','%Y-%m-%d'))

dated_slice.loc[inds, 'dt_del'] = dated_slice.loc[inds,'dt_obj'] - dated_slice.loc[inds,'dt_ns']

fr, b = np.histogram(dated_slice.loc[inds,'dt_del'].dt.days.values, bins=20)
hist_out = {
	'fr':fr,
	'b':b[:-1],
}
pd.DataFrame(hist_out).to_csv(os.path.join(os.getcwd(),'makefigs','data','fig-A9.csv'))

fig, ax = plt.subplots(1,1,figsize=(6,6))
dated_slice['dt_del'].dt.days.hist(bins=20, ax=ax, density=True, histtype='step',color='k', lw=3)
ax.set_yticklabels([])
ax.grid(False)
ax.set_ylabel('Freq')
ax.axvline(dated_slice['dt_del'].dt.days.mean(),c='r')
ax.axvline(dated_slice['dt_del'].dt.days.median(),c='r', ls=':')
ax.text(-850,0.0035,f'mean:    {dated_slice["dt_del"].dt.days.mean():.2f}\nmedian: {dated_slice["dt_del"].dt.days.median():.2f}\nstd:        {dated_slice["dt_del"].dt.days.std():.0f}', c='r')
ax.set_xlabel('Installation Date Lag (days)')
plt.savefig(os.path.join(root,'makefigs','figures','fig-A9_install_date_US.png'))
plt.show()