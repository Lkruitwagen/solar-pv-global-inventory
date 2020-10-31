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


class Figure:

    def __init__(self, make_arr=False):

        logger.info(f'initialising...')

        self.classes = ['cropland','urban_areas','treecover','grasslands','shrub/herbaceous/sparse','desert','barren_areas','wetlands','other']
        self.leg_keys = ['Cropland','Built-up Areas','Forest','Grassland','Shrubland','Aridlands','Barren Land','Wetlands','Other']
        self.bins = np.array([0.01,0.1,0.5,1,2,5,50,10000])*1000 # kW


        self.colors_dict = {
            'treecover':'#6ca966',
            'grasslands':'#d6d374',
            'shrub/herbaceous/sparse':'#3a5e2a',
            'desert':'#94130a',
            'wetlands':'#59cfc5',
            'cropland':'#ae7229',
            'urban_areas':'#8e3f99',
            'barren_areas':'#5a757d',
            'other':'#696969'
        }

        class_map = json.load(open(os.path.join(os.getcwd(),'data','lc300_classes.json'),'r'))


        logger.info(f'loading features...')

        gdf = gpd.read_file(os.path.join(os.getcwd(),'data','SPV_newmw.gpkg'))

        if make_arr:
            gdf['pt'] = gdf.geometry.representative_point()
            gdf['x'] = gdf['pt'].apply(lambda el: el.x)
            gdf['y'] = gdf['pt'].apply(lambda el: el.y)



        logger.info(f'mapping dates...')

        gdf['install_date'] = gdf['install_date'].str.replace('<2016-06','2000-01-01')
        gdf['install_date'] = gdf['install_date'].str.replace(',','')
        gdf['install_date'] = gdf['install_date'].str[0:10]

        gdf['dt_obj'] = pd.to_datetime(gdf['install_date'])



        if make_arr:
            df = pd.DataFrame(gdf[['capacity_mw','area','iso-3166-1','dt_obj','lc_vis','x','y']])
        else:
            df = pd.DataFrame(gdf[['capacity_mw','area','iso-3166-1','dt_obj','lc_vis']])

        logger.info(f'Cast to dummies')
        df = df.merge(pd.get_dummies(df['lc_vis']), how='left', left_index=True, right_index=True)
        self.df = df[df.area>10^1]
        print ('last')
        print (df['lc_vis'].unique())

        if make_arr:
            self.make_arr()

        self.df_lcpix = pd.read_csv(os.path.join(os.getcwd(),'data','LC300_latlonpixiso2.csv'))
        self.df_lcpv =  pd.read_csv(os.path.join(os.getcwd(),'data','LC300_pvpixiso2.csv'))
        self.df_lcworld =  pd.read_csv(os.path.join(os.getcwd(),'data','LC300_world.csv')).reset_index().rename(columns={'Unnamed: 0':'ISO_A2'})

        def map_codes(subdf, class_map):
            for kk,vv in class_map.items():
                subdf[kk] = subdf[vv].sum(axis=1)
            return subdf


        self.df_lcworld = map_codes(self.df_lcworld, class_map)
        self.df_lcpix = map_codes(self.df_lcpix, class_map)



    def make_global(self):
        
        df_ndt = self.df[self.df['dt_obj'].isna()]

        df = self.df.sort_values('dt_obj', na_position='first')

        arr = np.load(os.path.join(os.getcwd(),'data','lc300_arr.npz'))['arr']

        draw_arr = np.argmax(block_reduce(arr, (5,5,1), np.sum), axis=-1).astype(float)

        mask = block_reduce(arr, (5,5,1), np.sum).sum(axis=-1)==0

        draw_arr[mask]=np.nan

        cmap = colors.ListedColormap([self.colors_dict[kk] for kk in self.classes]) # cropland, urban areas, treecover,grasslands, shrub, desert, barren, wetlands, other
        bounds=[0,1,2,3,4,5,6,7,8]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        ne = gpd.read_file(os.path.join(os.getcwd(),'data','ne_10m_countries.gpkg'))



        for cc in self.classes:
            df[cc+'_orig'] = df[cc]
            df[cc] = df[cc]*df['capacity_mw']

            df_ndt[cc+'_orig'] = df_ndt[cc]
            df_ndt[cc] = df_ndt[cc]*df_ndt['capacity_mw']

        df = df.merge(df[self.classes].cumsum()/1000, how='left',left_index=True, right_index=True, suffixes=('','_cumsum'))


        #df = df.rename(columns={'forestshrub':'forest', 'grassy':'grasslands', 'human':'developed'})



        df['colors'] = df['lc_vis'].apply(lambda x: self.colors_dict[x])


        fig = plt.figure(figsize=(21,14))

        gs = GridSpec(4, 12, figure=fig, wspace=0.3, hspace=0.2)

        axs = {}

        axs['map'] = fig.add_subplot(gs[0:3,:], projection=ccrs.PlateCarree())
        axs['capmw'] = fig.add_subplot(gs[3,0:4])
        axs['hist'] = fig.add_subplot(gs[3,4:8])
        axs['local_skew'] = fig.add_subplot(gs[3,8])
        axs['global_skew'] = fig.add_subplot(gs[3,9])
        #axs['annotate'] = fig.add_subplot(gs[3,10])
        axs['legend'] = fig.add_subplot(gs[3,11:])
        #axs['legend'] = fig.add_axes([0.833,0,1.0,0.25])

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


        df.set_index('dt_obj')[[cc+'_cumsum' for cc in self.classes]].clip(0).plot.area(ax=axs['capmw'], color=[self.colors_dict[kk] for kk in self.classes],legend=False, lw=0)

        axs['capmw'].set_xticks([(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)) for mm in range(0,33,3)])
        axs['capmw'].set_xlabel('')
        axs['capmw'].set_xlim(['2016-06-01','2018-10-31'])
        axs['capmw'].set_ylabel('Est. Generating Capacity $[GW]$')
        axs['capmw'].set_xticklabels([str((dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)))[0:7] for mm in range(0,33,3)])

 

        for col in self.classes:
            np.log10(df[df[col+'_orig']>0].capacity_mw *1000).hist(ax=axs['hist'], bins=np.log10(self.bins), alpha=0.75, edgecolor=self.colors_dict[col],histtype='step', linewidth=3,density=True, fill=False)
        
        axs['hist'].grid(False)
        axs['hist'].set_yticks([])
        axs['hist'].set_xlabel('Est. Generating Capacity [$kW$]')
        axs['hist'].set_ylabel('Freq')
        axs['hist'].set_xlim([1, 6])
        axs['hist'].set_xticks([ii for ii in range(1,7)])
        axs['hist'].set_xticklabels(['10$^{}$'.format(ii) for ii in range(1,7)])
        axs['hist'].axvline(np.log10(1000), color='#363636', lw=1, linestyle='--')
        axs['hist'].axvline(np.log10(5000), color='#363636', lw=1, linestyle='--')
        axs['hist'].text(np.log10(1010),0.10,' 1MW',ha='left',fontsize=10, color='#363636')
        axs['hist'].text(np.log10(5100),0.55,' 5MW', fontsize=10, color='#363636')


        ## skew charts
        tot_pix = self.df_lcpix[self.classes].sum().sum()
        tot_pv = self.df_lcpv[self.classes].sum().sum()
        tot_world = self.df_lcworld[self.classes].sum().sum()

        diff = self.df_lcpv[self.classes].sum()/tot_pv \
            - self.df_lcpix[self.classes].sum()/tot_pix 

        print ('diff pv')
        print (diff)

        diff.plot.barh(ax=axs['local_skew'], color=[self.colors_dict[kk] for kk in self.classes])
        axs['local_skew'].axvline(0, color='k', lw=0.5)
        axs['local_skew'].spines['right'].set_visible(False)
        axs['local_skew'].spines['top'].set_visible(False)
        axs['local_skew'].spines['left'].set_visible(False)
        axs['local_skew'].set_yticks([])
        axs['local_skew'].set_yticklabels([])
        axs['local_skew'].set_xlim([-.2,.2])
        axs['local_skew'].set_xticks([-.2,0,.2])
        axs['local_skew'].set_xticklabels(['-20%','0%','20%'])


        diff =  self.df_lcpix[self.classes].sum()/tot_pix \
            - self.df_lcworld[self.classes].sum()/tot_world

        print ('diff pix')
        print (diff)

        diff.plot.barh(ax=axs['global_skew'], color=[self.colors_dict[kk] for kk in self.classes])
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


        

        leg_keys = self.leg_keys
            
        custom_lines = []
        for kk in self.classes:
            custom_lines.append(Line2D([0],[0],color=self.colors_dict[kk],marker='.',linestyle=None, lw=0, markersize=14))


        axs['legend'].axis('off')
        axs['legend'].legend(custom_lines, leg_keys, ncol=1, loc='center', handletextpad=0.1, columnspacing=0.1, fontsize=14, frameon=False)

        plt.annotate('Anthromes',horizontalalignment='right', xy=(1747, 1400-913), xycoords='figure pixels', fontsize=14) #fontweight='bold'
        plt.annotate('Biomes',horizontalalignment='right', xy=(1747, 1400-973), xycoords='figure pixels', fontsize=14)

        plt.annotate("",
            xy=(1754,1400-892), xycoords='figure pixels',
            xytext=(1754,1400-935), textcoords='figure pixels',
            arrowprops=dict(width=0.5, headwidth=0.5,facecolor='k',headlength=0.01,shrink=0.01),
            )

        plt.annotate("",
            xy=(1754,1400-951), xycoords='figure pixels',
            xytext=(1754,1400-1137), textcoords='figure pixels',
            arrowprops=dict(width=0.5, headwidth=0.5,facecolor='k',headlength=0.01,shrink=0.01),
            )

        plt.savefig(os.path.join(os.getcwd(),'makefigs','figures','fig-3_land_cover_global.png'))
        plt.show()





    def make_regional(self):

        countries_list = ['CN','US','IN','JP','DE','IT','ES','GB','TR','FR','CL','ZA','MX','TH','AU','KR','CA','CZ','GR','RO']
        upper_lim = {'CN':200,
                    'US':75,
                    'IN':50,
                    'JP':25,
                    'DE':25,
                    'IT':25,
                    'ES':15,
                    'GB':15,
                    'TR':10,
                    'FR':10,
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

        for cc in self.classes:
            df[cc+'_orig'] = df[cc]
            df[cc] = df[cc]*df['capacity_mw']

            df_ndt[cc+'_orig'] = df_ndt[cc]
            df_ndt[cc] = df_ndt[cc]*df_ndt['capacity_mw']


        #df['ts'] = df['dt_obj'].astype(np.int64)

        df['colors'] = df['lc_vis'].apply(lambda x: self.colors_dict[x])


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
            df_slice = df_slice.merge(df_slice[self.classes].cumsum()/1000, how='left',left_index=True, right_index=True, suffixes=('','_cumsum'))
            #df_slice = df_slice.rename(columns={'forestshrub':'forest', 'grassy':'grasslands', 'human':'developed'})

            df_slice = pd.concat([df_slice.iloc[[0],:],df_slice, df_slice.iloc[[-1],:]])
            df_slice.iloc[0,df.columns.get_loc('dt_obj')] = dt.strptime('2016-01-01','%Y-%m-%d')
            df_slice.iloc[-1,df.columns.get_loc('dt_obj')] = dt.strptime('2018-12-31','%Y-%m-%d')



            df_slice.set_index('dt_obj')[[cc+'_cumsum' for cc in self.classes]].clip(0).plot.area(ax=axs[country]['capmw'], color=[self.colors_dict[kk] for kk in self.classes],legend=False, lw=0)

            #axs['scatter'].set_xticks([np.datetime64(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)).astype(np.int64)*1000 for mm in range(0,33,3)])
            axs[country]['capmw'].set_xticks([(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)) for mm in range(0,33,6)])
            axs[country]['capmw'].set_xlabel('')
            axs[country]['capmw'].set_xlim(['2016-06-01','2018-10-31'])
            axs[country]['capmw'].set_xticklabels([])
            axs[country]['capmw'].set_ylabel('$[GW]$', labelpad=0.5)
            #axs[country]['capmw'].set_yscale('symlog')
            axs[country]['capmw'].set_ylim([0,upper_lim[country]])

            axs[country]['capmw'].text(0.02,0.8,str(country),transform=axs[country]['capmw'].transAxes,fontdict={'weight':'bold'})
            

 

            for col in self.classes:

                if (df_slice.loc[df_slice[col+'_orig']>0,'capacity_mw'].sum()/df_slice[self.classes].sum().sum())>0.05: # if its more than 5%
                    np.log10(df_slice[df_slice[col+'_orig']>0].capacity_mw *1000).hist(ax=axs[country]['hist'], bins=np.log10(self.bins), alpha=0.75, edgecolor=self.colors_dict[col],histtype='step', linewidth=3,density=True, fill=False)


            axs[country]['hist'].grid(False)
            axs[country]['hist'].set_yticks([])
            
            axs[country]['hist'].set_ylabel('Freq', labelpad=0.5)
            axs[country]['hist'].set_xlim([1, 5])
            axs[country]['hist'].set_xticks([ii for ii in range(1,7)])
            axs[country]['hist'].set_xticklabels([])
            axs[country]['hist'].axvline(np.log10(1000), color='#363636', lw=1, linestyle='--')
            axs[country]['hist'].axvline(np.log10(5000), color='#363636', lw=1, linestyle='--')

            ### skew charts
            tot_pix = self.df_lcpix.loc[self.df_lcpix['iso2']==country,self.classes].sum().sum()
            tot_pv = self.df_lcpv.loc[self.df_lcpv['iso2']==country,self.classes].sum().sum()
            tot_world = self.df_lcworld.loc[self.df_lcworld['ISO_A2']==country,self.classes].sum().sum()

            diff = self.df_lcpv.loc[self.df_lcpv['iso2']==country,self.classes].sum()/tot_pv \
                    - self.df_lcpix.loc[self.df_lcpix['iso2']==country,self.classes].sum()/tot_pix 



            print (country, diff.max(), diff.min())

            diff.plot.barh(ax=axs[country]['local_skew'], color=[self.colors_dict[cc] for cc in self.classes])
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


            diff =  self.df_lcpix.loc[self.df_lcpix['iso2']==country,self.classes].sum()/tot_pix \
                    - self.df_lcworld.loc[self.df_lcworld['ISO_A2']==country,self.classes].sum()/tot_world

            print (country, diff.max(), diff.min())

            diff.plot.barh(ax=axs[country]['country_skew'], color=[self.colors_dict[cc] for cc in self.classes])
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
            axs[country]['hist'].set_xticklabels(['10$^{}$'.format(ii) for ii in range(1,7)], fontsize=10)
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

        axs['GR']['hist'].text(np.log10(1010),0.7,' 1MW', fontsize=6, color='#363636')
        axs['GR']['hist'].text(np.log10(5100),0.7,' 5MW', fontsize=6, color='#363636')
        axs['RO']['hist'].text(np.log10(1010),0.1,' 1MW', fontsize=6, color='#363636')
        axs['RO']['hist'].text(np.log10(5100),0.75,' 5MW', fontsize=6, color='#363636')
        
        leg_keys = self.leg_keys

            
        custom_lines = []
        for kk in self.classes:
            custom_lines.append(Line2D([0],[0],color=self.colors_dict[kk],marker='.',linestyle=None, lw=0, markersize=13))

        axs['legend'].legend(custom_lines, leg_keys, ncol=len(self.classes), loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=False, shadow=False, frameon=False)

        #axs['legend'].axis('off')
        #axs['legend'].legend(custom_lines, list(self.colors_dict.keys()), ncol=2, loc='center', handletextpad=0.1, fontsize=18)
        
        plt.annotate('Anthromes',horizontalalignment='left', xy=(262, 1600-1389), xycoords='figure pixels', fontsize=14) #fontweight='bold'
        plt.annotate('Biomes',horizontalalignment='left', xy=(542, 1600-1389), xycoords='figure pixels', fontsize=14)

        plt.annotate("",
            xy=(262,1600-1395), xycoords='figure pixels',
            xytext=(506,1600-1395), textcoords='figure pixels',
            arrowprops=dict(width=0.5, headwidth=0.5,facecolor='k',headlength=0.001,shrink=0.001),
            )

        plt.annotate("",
            xy=(542,1600-1395), xycoords='figure pixels',
            xytext=(1390,1600-1395), textcoords='figure pixels',
            arrowprops=dict(width=0.5, headwidth=0.5,facecolor='k',headlength=0.001,shrink=0.001),
            )

        plt.savefig(os.path.join(os.getcwd(),'makefigs','figures','fig-A10_land_cover_regional.png'))
        plt.show()

    def make_arr(self):

        arr = np.zeros((360*10,180*10,len(self.classes)))

        ind_dict = dict(zip(self.classes,range(len(self.classes))))

        for ii_r,row in enumerate(self.df.iterrows()):
            if ii_r % 1000 ==0:
                print ('ii_r')
                print (row)
            x = int((row[1]['x'] + 180)*10)
            y = int((row[1]['y'] + 90)*10)

            if row[1]['lc_vis'] in self.classes:

                arr[x,y,ind_dict[row[1]['lc_vis']]] += row[1]['area']

        print ('writing arr')
        np.savez(os.path.join(os.getcwd(),'data','lc300_arr.npz'),arr=arr)

        print ('summarising df...')
        x,y = np.where(arr.sum(axis=-1)>0)

        records = {}
        for pix_x, pix_y in list(zip(x,y)):
            records[(pix_x,pix_y)]= dict(zip(self.classes,arr[pix_x,pix_y,:].tolist()))

        df_arr = pd.DataFrame.from_dict(records).T

        df_arr.to_csv(os.path.join(os.getcwd(),'data','LC300_pvpix.csv'))

        for bin_bounds in [(0,1),(1,5),(5,4000)]:

            print ('doing bin bounds:', bin_bounds)

            arr = np.zeros((360*10,180*10,len(self.classes)))

            ind_dict = dict(zip(self.classes,range(len(self.classes))))

            for ii_r,row in enumerate(self.df.loc[(self.df['capacity_mw']>bin_bounds[0])&(self.df['capacity_mw']<=bin_bounds[1])].iterrows()):
                if ii_r % 1000 ==0:
                    print ('ii_r')
                    print (row)
                x = int((row[1]['x'] + 180)*10)
                y = int((row[1]['y'] + 90)*10)

                if row[1]['lc_vis'] in self.classes:

                    arr[x,y,ind_dict[row[1]['lc_vis']]] += row[1]['area']

            print ('writing arr')
            np.savez(os.path.join(os.getcwd(),'data',f'lc300_arr_{bin_bounds[0]}.npz'),arr=arr)

            print ('summarising df...')
            x,y = np.where(arr.sum(axis=-1)>0)

            records = {}
            for pix_x, pix_y in list(zip(x,y)):
                records[(pix_x,pix_y)]= dict(zip(self.classes,arr[pix_x,pix_y,:].tolist()))

            df_arr = pd.DataFrame.from_dict(records).T

            df_arr.to_csv(os.path.join(os.getcwd(),'data',f'LC300_pvpix_{bin_bounds[0]}.csv'))




if __name__=="__main__":
    generator=Figure(make_arr=False)
    #generator.make_arr()
    #generator.make_global()
    generator.make_regional()
