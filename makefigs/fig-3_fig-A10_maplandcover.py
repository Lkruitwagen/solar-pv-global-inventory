import pickle, logging, os, sys, json
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import geometry
from skimage.measure import block_reduce

import cartopy.crs as ccrs
import cartopy.feature as cfeature

gpd.options.use_pygeos=False

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from matplotlib import colors

from scipy.ndimage.filters import gaussian_filter
from skimage import exposure

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logger = logging.getLogger(__name__)

""" original color scheme
        self.colors_dict = {
            'forest':'#2f9149',
            'grasslands':'#b6eb7f',
            'wetlands':'#7fe7eb',
            'cropland':'#ed8540',
            'developed':'#d000ff',
            'barren':'#0000ff',
            'other':'#636363'
        }

        '#ed8540', '#2f9149','#b6eb7f','#d000ff','cyan','b'  #crop, forest, grass, dev, wet, barren
        '#ae7229', '#6ca966', '#edeccd','#f10100','#71a4c1','#b3afa4'  # new color scheme -> replace barren: 42798a
"""


class Figure:

    def __init__(self, make_arr=False):

        logger.info(f'initialising...')

        self.colors_dict = {
            'forest':'#6ca966',
            'grasslands':'#edeccd',
            'wetlands':'#71a4c1',
            'cropland':'#ae7229',
            'developed':'#f10100',
            'barren':'#42798a',
            'other':'#e7effc'
        }

        self.colors_base = {
            'forestshrub':'#6ca966',
            'grassy':'#edeccd',
            'wetlands':'#71a4c1',
            'cropland':'#ae7229',
            'human':'#f10100',
            'barren':'#42798a',
            'other':'#e7effc',
            np.nan:'#e7effc'
        }

        self.corine_countries = ['AL', 'AT', 'BE', 'BA', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IS', 'IE', 'IT', 'XK', 'LV', 'LI', 'LT', 'LU', 'MK', 'MT', 'ME', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SK', 'SI', 'ES', 'SE', 'CH', 'TR', 'GB']

        logger.info(f'loading features...')

        landcover_fts = json.load(open(os.path.join(os.getcwd(), 'data','ABCD_finalized.geojson'),'r'))['features']

        if make_arr:
            records = []
            for ft in landcover_fts:
                rec = ft['properties']
                pt = geometry.shape(ft['geometry']).representative_point()
                rec['x'] = pt.x
                rec['y'] = pt.y
                records.append(rec)

        if make_arr:
            df = pd.DataFrame.from_records(records)
        else:
            df = pd.DataFrame.from_records([ft['properties'] for ft in landcover_fts])



        logger.info(f'mapping landcover to classes...')

        df['install_date'] = df['install_date'].str.replace('<2016-06','2000-01-01')
        df['install_date'] = df['install_date'].str.replace(',','')
        df['install_date'] = df['install_date'].str[0:10]

        df['dt_obj'] = pd.to_datetime(df['install_date'])


        """  If needing to remake the labels
        labels = {}
        labels['CORINE'] = pickle.load(open(os.path.join(os.getcwd(),'data','class_labels_CORINE.pkl'),'rb'))
        labels['MODIS'] = pickle.load(open(os.path.join(os.getcwd(),'data','class_labels_MODIS.pkl'),'rb'))
        labels['CDL'] = pickle.load(open(os.path.join(os.getcwd(),'data','class_labels_cdl.pkl'),'rb'))

        labels_agg = {}
        labels_agg['CORINE'] = {
            'forestshrub':[23,24,25,29],
            'wetlands':[35,36,37,38],
            'human':[1,2,3,4,5,6,7,8,9,10,11],
            'cropland':[12,13,14,15,16,17,18,19,20,21,22],
            'grassy':[26,27,28,],
            'other':[30,31,32,33,34,39,40,41,42,43,44,45,46,47],
        }

        labels_agg['MODIS'] = {    
            'forestshrub':[1,2,3,4,5,6,7,8],
            'wetlands':[11],
            'human':[13],
            'cropland':[12,14],
            'grassy':[9,10],
            'other':[15,16,17],
        }

        labels_agg['CDL'] = {    
            'forestshrub':[63,64, 141, 142, 143, 152],
            'wetlands':[87, 190, 195],
            'human':[82, 121, 122, 123, 124],
            'cropland':[], # else
            'grassy':[59,60,61,62, 176],
            'other':[0, 65, 81, 83, 88, 111, 112, 131], #''
        }

        existing_labels = [el for kk,vv in labels_agg['CDL'].items() for el in vv]

        for kk, vv in labels['CDL'].items():
            if kk not in existing_labels:
                if vv=='':
                    labels_agg['CDL']['other'].append(kk)
                else:
                    labels_agg['CDL']['cropland'].append(kk)
        """

        labels_agg = json.load(open(os.path.join(os.getcwd(),'data','all_labels.json'),'r'))

        df_cdl = pd.DataFrame.from_dict(labels_agg['CDL'], orient='index')
        df_cdl = df_cdl.unstack().dropna().reset_index().set_index(0).drop(columns=['level_0']).sort_index()
        df_cdl.index = df_cdl.index.astype(int).astype(str)

        df_corine = pd.DataFrame.from_dict(labels_agg['CORINE'], orient='index')
        df_corine = df_corine.unstack().dropna().reset_index().set_index(0).drop(columns=['level_0']).sort_index()
        df_corine.index = df_corine.index.astype(int).astype(str)

        df_modis = pd.DataFrame.from_dict(labels_agg['MODIS'], orient='index')
        df_modis = df_modis.unstack().dropna().reset_index().set_index(0).drop(columns=['level_0']).sort_index()
        df_modis.index = df_modis.index.astype(int).astype(str)


        df['land_cover_vis'] = ''

        # US
        df.loc[(df['iso-3166-1']=='US') & (df['dt_obj']>dt.strptime('2016-06-01','%Y-%m-%d')),'land_cover_vis'] = df.loc[((df['iso-3166-1']=='US')& (df['dt_obj']>dt.strptime('2016-06-01','%Y-%m-%d'))),'land_cover_CDL_2012'].map(df_cdl.to_dict()['level_1'])
        df.loc[(df['iso-3166-1']=='US') & ~(df['dt_obj']>dt.strptime('2016-06-01','%Y-%m-%d')),'land_cover_vis'] = df.loc[((df['iso-3166-1']=='US')& ~(df['dt_obj']>dt.strptime('2016-06-01','%Y-%m-%d'))),'land_cover_CDL_2009'].map(df_cdl.to_dict()['level_1'])


        # EEA
        df.loc[(df['iso-3166-1'].isin(self.corine_countries) &  (df['dt_obj']>dt.strptime('2016-06-01','%Y-%m-%d'))),'land_cover_vis'] = df.loc[(df['iso-3166-1'].isin(self.corine_countries) &  (df['dt_obj']>dt.strptime('2016-06-01','%Y-%m-%d'))),'land_cover_CORINE_2012'].map(df_corine.to_dict()['level_1'])
        df.loc[(df['iso-3166-1'].isin(self.corine_countries) &  ~(df['dt_obj']>dt.strptime('2016-06-01','%Y-%m-%d'))),'land_cover_vis'] = df.loc[(df['iso-3166-1'].isin(self.corine_countries) &  ~(df['dt_obj']>dt.strptime('2016-06-01','%Y-%m-%d'))),'land_cover_CORINE_2006'].map(df_corine.to_dict()['level_1'])
        

        # Else
        df.loc[(~df['iso-3166-1'].isin(self.corine_countries+['US']) & (df['dt_obj']>dt.strptime('2016-06-01','%Y-%m-%d'))) ,'land_cover_vis'] = df.loc[(~df['iso-3166-1'].isin(self.corine_countries+['US']) & (df['dt_obj']>dt.strptime('2016-06-01','%Y-%m-%d'))) ,'land_cover_MODIS_2012'].map(df_modis.to_dict()['level_1'])
        df.loc[(~df['iso-3166-1'].isin(self.corine_countries+['US']) & ~(df['dt_obj']>dt.strptime('2016-06-01','%Y-%m-%d'))) ,'land_cover_vis'] = df.loc[(~df['iso-3166-1'].isin(self.corine_countries+['US']) & ~(df['dt_obj']>dt.strptime('2016-06-01','%Y-%m-%d'))) ,'land_cover_MODIS_2007'].map(df_modis.to_dict()['level_1'])


        if make_arr:
            df = df[['capacity_mw','area','iso-3166-1','dt_obj','land_cover_vis','x','y']]
        else:
            df = df[['capacity_mw','area','iso-3166-1','dt_obj','land_cover_vis']]
        df = df.merge(pd.get_dummies(df['land_cover_vis']), how='left', left_index=True, right_index=True)
        self.df = df[df.area>10^1]
        print ('last')
        print (df.land_cover_vis.unique())

        if make_arr:
            self.make_arr()

        self.df_lcpix = pd.read_csv(os.path.join(os.getcwd(),'data','landcover_latlonpix.csv'))
        self.df_lcpv =  pd.read_csv(os.path.join(os.getcwd(),'data','landcover_pvpix.csv'))
        self.df_lcworld =  pd.read_csv(os.path.join(os.getcwd(),'data','landcover_world.csv'))


    def make_global(self):
        
        df_ndt = self.df[self.df['dt_obj'].isna()]

        df = self.df.sort_values('dt_obj', na_position='first')

        arr = np.load(os.path.join(os.getcwd(),'data','land_cover_arr.npz'))['arr']

        draw_arr = np.argmax(block_reduce(arr, (5,5,1), np.sum), axis=-1).astype(float)

        mask = block_reduce(arr, (5,5,1), np.sum).sum(axis=-1)==0

        draw_arr[mask]=np.nan

        cmap = colors.ListedColormap(['#ae7229', '#6ca966', '#edeccd','#f10100','#71a4c1','#42798a']) # cropland, forest, grassland, developed, wetlands, barren/other
        bounds=[0,1,2,3,4,5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        ne = gpd.read_file(os.path.join(os.getcwd(),'data','ne_10m_countries.gpkg'))



        for cc in ['cropland', 'forestshrub', 'grassy', 'human', 'barren','other', 'wetlands']:
            df[cc+'_orig'] = df[cc]
            df[cc] = df[cc]*df['capacity_mw']

            df_ndt[cc+'_orig'] = df_ndt[cc]
            df_ndt[cc] = df_ndt[cc]*df_ndt['capacity_mw']

        df = df.merge(df[['cropland','forestshrub','grassy','human','barren','other','wetlands']].cumsum()/1000, how='left',left_index=True, right_index=True, suffixes=('','_cumsum'))


        df = df.rename(columns={'forestshrub':'forest', 'grassy':'grasslands', 'human':'developed'})



        df['colors'] = df['land_cover_vis'].apply(lambda x: self.colors_base[x])


        fig = plt.figure(figsize=(21,14))

        gs = GridSpec(4, 12, figure=fig, wspace=0.3, hspace=0.2)

        axs = {}

        axs['map'] = fig.add_subplot(gs[0:3,:], projection=ccrs.PlateCarree())
        axs['capmw'] = fig.add_subplot(gs[3,0:4])
        axs['hist'] = fig.add_subplot(gs[3,4:8])
        axs['local_skew'] = fig.add_subplot(gs[3,8])
        axs['global_skew'] = fig.add_subplot(gs[3,9])
        axs['legend'] = fig.add_subplot(gs[3,10:])

        axs['map'].set_title('(a)')
        axs['capmw'].set_title('(b)')
        axs['hist'].set_title('(c)')
        axs['local_skew'].set_title('(d)')
        axs['global_skew'].set_title('(e)')


        im = np.flip(plt.imread(os.path.join(os.getcwd(),'data','GRAY_LR_SR_W.tif')),axis=0)
        im = exposure.adjust_gamma(gaussian_filter(im,7),0.2)/256
        axs['map'].imshow(im,extent=[-180,180,-90,90], transform=ccrs.PlateCarree(),interpolation='nearest', origin='lower',cmap='Greys',vmin=0.85, vmax=1.1, zorder=1)


        ne.geometry.plot(ax=axs['map'],color='#e6e6e6', edgecolor=None, zorder=0)

        axs['map'].imshow(np.swapaxes(draw_arr,0,1),extent=[-180,180,-90,90], transform=ccrs.PlateCarree(),interpolation='nearest', origin='lower',cmap=cmap,norm=norm, zorder=2)
        

        bbox = geometry.box(-180,-90,180,90)
        world_mp = pickle.load(open(os.path.join(os.getcwd(),'data','world_mp.pkl'),'rb'))
        inv = bbox.difference(world_mp)

        inv = gpd.GeoDataFrame(pd.DataFrame(inv,index=list(range(len(inv))),columns=['geometry']),geometry='geometry', crs={'init':'epsg:4326'})

        inv.geometry.plot(ax=axs['map'],color='white', edgecolor=None,zorder=3)



        ne.geometry.boundary.plot(ax=axs['map'],color=None,edgecolor='#616161',linewidth=1,zorder=4)


        axs['map'].set_ylim([-60,85])


        df.set_index('dt_obj')[['cropland_cumsum', 'forestshrub_cumsum', 'grassy_cumsum', 'wetlands_cumsum', 'human_cumsum','barren_cumsum', 'other_cumsum']].clip(0).plot.area(ax=axs['capmw'], color=[self.colors_dict[kk] for kk in ['cropland', 'forest', 'grasslands', 'wetlands','developed', 'barren','other']],legend=False, lw=0)

        axs['capmw'].set_xticks([(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)) for mm in range(0,33,3)])
        axs['capmw'].set_xlabel('')
        axs['capmw'].set_xlim(['2016-06-01','2018-10-31'])
        axs['capmw'].set_ylabel('Est. Generating Capacity $[GW]$')
        axs['capmw'].set_xticklabels([str((dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)))[0:7] for mm in range(0,33,3)])

 

        for col in ['cropland','forestshrub','grassy','wetlands','human','barren','other']:
            np.log10(df[df[col+'_orig']>0].capacity_mw *1000).hist(ax=axs['hist'], bins=10, alpha=0.75, edgecolor=self.colors_base[col],histtype='step', linewidth=3,density=True, fill=False)
        
        axs['hist'].grid(False)
        axs['hist'].set_yticks([])
        axs['hist'].set_xlabel('Est. Generating Capacity [$kW$]')
        axs['hist'].set_ylabel('Freq')
        axs['hist'].set_xlim([1, 7])
        axs['hist'].set_xticks([ii for ii in range(1,8)])
        axs['hist'].set_xticklabels(['10$^{}$'.format(ii) for ii in range(1,8)])


        ## skew charts
        tot_pix = self.df_lcpix[['cropland','forestshrub','grassy','human','wetlands','barren']].sum().sum()
        tot_pv = self.df_lcpv[['cropland','forestshrub','grassy','human','wetlands','barren']].sum().sum()
        tot_world = self.df_lcworld[['cropland','forestshrub','grassy','human','wetlands','barren']].sum().sum()

        diff = self.df_lcpv[['cropland','forestshrub','grassy','human','wetlands','barren']].sum()/tot_pv \
            - self.df_lcpix[['cropland','forestshrub','grassy','human','wetlands','barren']].sum()/tot_pix 

        print ('diff pv')
        print (diff)

        diff.plot.barh(ax=axs['local_skew'], color=['#ae7229', '#6ca966', '#edeccd','#f10100','#71a4c1','#42798a'])
        axs['local_skew'].axvline(0, color='k', lw=0.5)
        axs['local_skew'].spines['right'].set_visible(False)
        axs['local_skew'].spines['top'].set_visible(False)
        axs['local_skew'].spines['left'].set_visible(False)
        axs['local_skew'].set_yticks([])
        axs['local_skew'].set_yticklabels([])
        axs['local_skew'].set_xlim([-.2,.2])
        axs['local_skew'].set_xticks([-.2,0,.2])
        axs['local_skew'].set_xticklabels(['-20%','0%','20%'])


        diff =  self.df_lcpix[['cropland','forestshrub','grassy','human','wetlands','barren']].sum()/tot_pix \
            - self.df_lcworld[['cropland','forestshrub','grassy','human','wetlands','barren']].sum()/tot_world

        print ('diff pix')
        print (diff)

        diff.plot.barh(ax=axs['global_skew'], color=['#ae7229', '#6ca966', '#edeccd','#f10100','#71a4c1','#42798a'])
        axs['global_skew'].axvline(0, color='k', lw=0.5)
        axs['global_skew'].spines['right'].set_visible(False)
        axs['global_skew'].spines['top'].set_visible(False)
        axs['global_skew'].spines['left'].set_visible(False)
        axs['global_skew'].set_yticks([])
        axs['global_skew'].set_yticklabels([])
        axs['global_skew'].set_xlim([-.5,.5])
        axs['global_skew'].set_xticks([-0.5, 0, 0.5])
        axs['global_skew'].set_xticklabels(['-50%','0%','50%'])

        adj = {0:'left',1:'center',2:'right'}

        for ii_t, tick in enumerate(axs['local_skew'].xaxis.get_majorticklabels()):
            tick.set_horizontalalignment(adj[ii_t])
        for ii_t, tick in enumerate(axs['global_skew'].xaxis.get_majorticklabels()):
            tick.set_horizontalalignment(adj[ii_t])


        

        leg_keys = ['Forest','Grassland','Wetland','Cropland','Developed','Barren','Other']
            
        custom_lines = []
        for kk,vv in self.colors_dict.items():
            custom_lines.append(Line2D([0],[0],color=vv,marker='.',linestyle=None, lw=0, markersize=10))


        axs['legend'].axis('off')
        axs['legend'].legend(custom_lines, leg_keys, ncol=1, loc='center', handletextpad=0.1, columnspacing=0.1, fontsize=18, frameon=False)

        plt.savefig(os.path.join(os.getcwd(),'makefigs','figures','fig-3_land_cover_global.png'))
        plt.show()





    def make_regional(self):

        countries_list = ['CN','US','IN','JP','DE','IT','ES','GB','TR','FR','CL','ZA','MX','TH','AU','KR','CA','CZ','GR','RO']
        upper_lim = {'CN':150,
                    'US':50,
                    'IN':25,
                    'JP':25,
                    'DE':25,
                    'IT':25,
                    'ES':10,
                    'GB':10,
                    'TR':5,
                    'FR':5,
                    'CL':5,
                    'ZA':5,
                    'MX':5,
                    'TH':5,
                    'AU':5,
                    'KR':5,
                    'CA':5,
                    'CZ':5,
                    'AF':5,
                    'GR':5,
                    'RO':5}

        df_ndt = self.df[self.df['dt_obj'].isna()]
        #df = self.df[~self.df['dt_obj'].isna()].sort_values('dt_obj')  
        df = self.df.sort_values('dt_obj', na_position='first')

        for cc in ['cropland', 'forestshrub', 'grassy', 'human', 'barren','other', 'wetlands']:
            df[cc+'_orig'] = df[cc]
            df[cc] = df[cc]*df['capacity_mw']

            df_ndt[cc+'_orig'] = df_ndt[cc]
            df_ndt[cc] = df_ndt[cc]*df_ndt['capacity_mw']


        #df['ts'] = df['dt_obj'].astype(np.int64)

        df['colors'] = df['land_cover_vis'].apply(lambda x: self.colors_base[x])


        fig = plt.figure(figsize=(16,16))

        gs = GridSpec(int(len(countries_list)/2), 25, figure=fig, wspace=0.55, hspace=0.25)
        gs0 = GridSpec(1,1, figure=fig, wspace=0.3, hspace=0.2)

        axs={}

        axs['legend'] = fig.add_subplot(gs0[0,0])
        axs['legend'].axis('off')

        for ii_c,country in enumerate(countries_list):
            row=ii_c//2
            col = int(12*(ii_c%2)+ii_c%2) # [0,2] as appropriate

            axs[country]={}
            axs[country]['capmw'] = fig.add_subplot(gs[row,col:col+4])
            axs[country]['hist'] = fig.add_subplot(gs[row,col+4:col+8])
            axs[country]['local_skew'] = fig.add_subplot(gs[row,col+8:col+10])
            axs[country]['country_skew'] = fig.add_subplot(gs[row,col+10:col+12])
            


        for ii_c, country in enumerate(countries_list):

            df_slice = df[df['iso-3166-1']==country]
            df_slice = df_slice.merge(df_slice[['cropland','forestshrub','grassy','human','other','barren','wetlands']].cumsum()/1000, how='left',left_index=True, right_index=True, suffixes=('','_cumsum'))
            df_slice = df_slice.rename(columns={'forestshrub':'forest', 'grassy':'grasslands', 'human':'developed'})

            df_slice = pd.concat([df_slice.iloc[[0],:],df_slice, df_slice.iloc[[-1],:]])
            df_slice.iloc[0,df.columns.get_loc('dt_obj')] = dt.strptime('2016-01-01','%Y-%m-%d')
            df_slice.iloc[-1,df.columns.get_loc('dt_obj')] = dt.strptime('2018-12-31','%Y-%m-%d')



            df_slice.set_index('dt_obj')[['cropland_cumsum', 'forestshrub_cumsum', 'grassy_cumsum', 'wetlands_cumsum','barren_cumsum', 'human_cumsum', 'other_cumsum']].clip(0).plot.area(ax=axs[country]['capmw'], color=[self.colors_dict[kk] for kk in ['cropland', 'forest', 'grasslands', 'wetlands','barren','developed','other']],legend=False, lw=0)

            #axs['scatter'].set_xticks([np.datetime64(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)).astype(np.int64)*1000 for mm in range(0,33,3)])
            axs[country]['capmw'].set_xticks([(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)) for mm in range(0,33,6)])
            axs[country]['capmw'].set_xlabel('')
            axs[country]['capmw'].set_xlim(['2016-06-01','2018-10-31'])
            axs[country]['capmw'].set_xticklabels([])
            axs[country]['capmw'].set_ylabel('$[GW]$', labelpad=0.5)
            #axs[country]['capmw'].set_yscale('symlog')
            axs[country]['capmw'].set_ylim([0,upper_lim[country]])

            axs[country]['capmw'].text(0.02,0.8,str(country),transform=axs[country]['capmw'].transAxes,fontdict={'weight':'bold'})
            

 

            for col in ['cropland','forestshrub','grassy','wetlands','human','barren']:

                if (df_slice.loc[df_slice[col+'_orig']>0,'capacity_mw'].sum()/df_slice[['cropland','forest','grasslands','wetlands','developed','barren']].sum().sum())>0.05: # if its more than 5%
                    np.log10(df_slice[df_slice[col+'_orig']>0].capacity_mw *1000).hist(ax=axs[country]['hist'], bins=10, alpha=0.75, edgecolor=self.colors_base[col],histtype='step', linewidth=3,density=True, fill=False)
        
            axs[country]['hist'].grid(False)
            axs[country]['hist'].set_yticks([])
            
            axs[country]['hist'].set_ylabel('Freq', labelpad=0.5)
            axs[country]['hist'].set_xlim([1, 6])
            axs[country]['hist'].set_xticks([ii for ii in range(1,7)])
            axs[country]['hist'].set_xticklabels([])

            ### skew charts
            tot_pix = self.df_lcpix.loc[self.df_lcpix['iso2']==country,['cropland','forestshrub','grassy','human','wetlands','barren']].sum().sum()
            tot_pv = self.df_lcpv.loc[self.df_lcpv['iso2']==country,['cropland','forestshrub','grassy','human','wetlands','barren']].sum().sum()
            tot_world = self.df_lcworld.loc[self.df_lcworld['ISO_A2']==country,['cropland','forestshrub','grassy','human','wetlands','barren']].sum().sum()

            diff = self.df_lcpv.loc[self.df_lcpv['iso2']==country,['cropland','forestshrub','grassy','human','wetlands','barren']].sum()/tot_pv \
                    - self.df_lcpix.loc[self.df_lcpix['iso2']==country,['cropland','forestshrub','grassy','human','wetlands','barren']].sum()/tot_pix 



            print (country, diff.max(), diff.min())

            diff.plot.barh(ax=axs[country]['local_skew'], color=['#ae7229', '#6ca966', '#edeccd','#f10100','#71a4c1','#42798a'])
            axs[country]['local_skew'].axvline(0, color='k', lw=0.5)
            axs[country]['local_skew'].spines['right'].set_visible(False)
            axs[country]['local_skew'].spines['top'].set_visible(False)
            axs[country]['local_skew'].spines['left'].set_visible(False)
            axs[country]['local_skew'].set_yticks([])
            axs[country]['local_skew'].set_yticklabels([])
            #axs[country]['local_skew'].set_xscale('symlog')
            axs[country]['local_skew'].set_xlim([-0.25,0.25])
            axs[country]['local_skew'].set_xticklabels([])
            #ax.set_xlim([-.2,.2])
            #ax.set_xticks(np.arange(-.2,.2,0.05))

            for ii_v, (idx, val) in enumerate(diff.items()):
                if val>0.25:
                    axs[country]['local_skew'].text(-0.03, ii_v, '{:.0%}'.format(val), horizontalalignment='right', verticalalignment='center', fontsize=10)
                elif val<-0.25:
                    axs[country]['local_skew'].text(0.03, ii_v, '{:.0%}'.format(val), horizontalalignment='left', verticalalignment='center', fontsize=10)


            diff =  self.df_lcpix.loc[self.df_lcpix['iso2']==country,['cropland','forestshrub','grassy','human','wetlands','barren']].sum()/tot_pix \
                    - self.df_lcworld.loc[self.df_lcworld['ISO_A2']==country,['cropland','forestshrub','grassy','human','wetlands','barren']].sum()/tot_world

            print (country, diff.max(), diff.min())

            diff.plot.barh(ax=axs[country]['country_skew'], color=['#ae7229', '#6ca966', '#edeccd','#f10100','#71a4c1','#42798a'])
            axs[country]['country_skew'].axvline(0, color='k', lw=0.5)
            axs[country]['country_skew'].spines['right'].set_visible(False)
            axs[country]['country_skew'].spines['top'].set_visible(False)
            axs[country]['country_skew'].spines['left'].set_visible(False)
            axs[country]['country_skew'].set_yticks([])
            axs[country]['country_skew'].set_yticklabels([])
            #axs[country]['country_skew'].set_xscale('symlog')
            axs[country]['country_skew'].set_xlim([-0.25,0.25])
            axs[country]['country_skew'].set_xticklabels([])

            for ii_v, (idx, val) in enumerate(diff.items()):
                if val>0.25:
                    axs[country]['country_skew'].text(-0.03, ii_v, '{:.0%}'.format(val), horizontalalignment='right', verticalalignment='center', fontsize=10)
                elif val<-0.25:
                    axs[country]['country_skew'].text(0.03, ii_v, '{:.0%}'.format(val), horizontalalignment='left', verticalalignment='center', fontsize=10)


        adj = {0:'left',1:'center',2:'right'}

        for country in countries_list[-2:]:
            axs[country]['capmw'].set_xticks([(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)) for mm in range(0,33,6)])
            axs[country]['capmw'].set_xticklabels([str((dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)))[0:7] for mm in range(0,33,6)], fontsize=10, rotation='vertical', ha='center')
            axs[country]['hist'].set_xlabel('Est. Gen. Capacity [kW]')
            axs[country]['hist'].set_xticklabels(['10$^{}$'.format(ii) for ii in range(1,8)], fontsize=10)
            local_labels = axs[country]['local_skew'].set_xticklabels(['-25','0','25%'], fontsize=10)
            country_labels = axs[country]['country_skew'].set_xticklabels(['-25','0','25%'], fontsize=10)



            for ii_t, tick in enumerate(axs[country]['local_skew'].xaxis.get_majorticklabels()):
                tick.set_horizontalalignment(adj[ii_t])
            for ii_t, tick in enumerate(axs[country]['country_skew'].xaxis.get_majorticklabels()):
                tick.set_horizontalalignment(adj[ii_t])

        for country in countries_list[0:2]:
            axs[country]['capmw'].set_title('(a)')
            axs[country]['hist'].set_title('(b)')
            axs[country]['local_skew'].set_title('(c)')
            axs[country]['country_skew'].set_title('(d)')

        
        leg_keys = ['Forest','Grassland','Wetland','Cropland','Developed','Barren']

            
        custom_lines = []
        for kk,vv in self.colors_dict.items():
            custom_lines.append(Line2D([0],[0],color=vv,marker='.',linestyle=None, lw=0, markersize=10))

        axs['legend'].legend(custom_lines, leg_keys, ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.08), fancybox=False, shadow=False, frameon=False)

        #axs['legend'].axis('off')
        #axs['legend'].legend(custom_lines, list(self.colors_dict.keys()), ncol=2, loc='center', handletextpad=0.1, fontsize=18)
        

        plt.savefig(os.path.join(os.getcwd(),'makefigs','figures','fig-A10_land_cover_regional.png'))
        plt.show()

    def make_arr(self):

        arr = np.zeros((360*10,180*10,7))

        ind_dict = dict(zip(['cropland', 'forestshrub', 'grassy', 'human','wetlands', 'barren','other'],range(7)))

        for ii_r,row in enumerate(self.df.iterrows()):
            if ii_r % 1000 ==0:
                print ('ii_r')
                print (row)
            x = int((row[1]['x'] + 180)*10)
            y = int((row[1]['y'] + 90)*10)

            if row[1]['land_cover_vis'] in ['cropland', 'forestshrub', 'grassy', 'human','wetlands','barren', 'other', 'wetlands']:

                arr[x,y,ind_dict[row[1]['land_cover_vis']]] += row[1]['area']

        np.savez(os.path.join(os.getcwd(),'data','land_cover_arr.npz'),arr=arr)




if __name__=="__main__":
    generator=Figure(make_arr=False)
    #generator.make_arr()
    generator.make_global()
    #generator.make_regional()
