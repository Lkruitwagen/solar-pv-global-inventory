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
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase

from scipy.ndimage.filters import gaussian_filter
from skimage import exposure

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logger = logging.getLogger(__name__)


class Figure:

    def __init__(self, make_arr=False):

        logger.info(f'initialising...')


        self.corine_countries = ['AL', 'AT', 'BE', 'BA', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IS', 'IE', 'IT', 'XK', 'LV', 'LI', 'LT', 'LU', 'MK', 'MT', 'ME', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SK', 'SI', 'ES', 'SE', 'CH', 'TR', 'GB']
        self.params ={
        'agg':.5, #decimal degrees
        'minsize':1,
        'maxsize':100,
        'minmw':2,
        'maxmw':200,
        }


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


        self.df = df[df.area>10^1]


        #if make_arr:
        #    self.make_arr()


    def make_global(self):
        
        df_ndt = self.df[self.df['dt_obj'].isna()]
        #df = self.df[~self.df['dt_obj'].isna()].sort_values('dt_obj')  
        df = self.df.sort_values('dt_obj', na_position='first')




        


        ne = gpd.read_file(os.path.join(os.getcwd(),'data','ne_10m_countries.gpkg'))



        fig = plt.figure(figsize=(21,14))

        gs = GridSpec(4, 36, figure=fig, wspace=0.3, hspace=0.1)

        axs = {}

        axs['map'] = fig.add_subplot(gs[0:3,:], projection=ccrs.PlateCarree())
        axs['capmw_0'] = fig.add_subplot(gs[3,0:3])
        axs['capmw_1'] = fig.add_subplot(gs[3,4:15])
        axs['capmw_2'] = fig.add_subplot(gs[3,16:31])
        #axs['hist'] = fig.add_subplot(gs[3,5:10]) # maybe do a size distribution by time?
        axs['legend'] = fig.add_subplot(gs[3,32:])


        im = np.flip(plt.imread(os.path.join(os.getcwd(),'data','GRAY_LR_SR_W.tif')),axis=0)
        im = exposure.adjust_gamma(gaussian_filter(im,7),0.2)/256
        axs['map'].imshow(im,extent=[-180,180,-90,90], transform=ccrs.PlateCarree(),interpolation='nearest', origin='lower',cmap='Greys',vmin=0.85, vmax=1.1, zorder=1)


        ne.geometry.plot(ax=axs['map'],color='#e6e6e6', edgecolor=None, zorder=0)


        bbox = geometry.box(-180,-90,180,90)
        world_mp = pickle.load(open(os.path.join(os.getcwd(),'data','world_mp.pkl'),'rb'))
        inv = bbox.difference(world_mp)

        inv = gpd.GeoDataFrame(pd.DataFrame(inv,index=list(range(len(inv))),columns=['geometry']),geometry='geometry', crs={'init':'epsg:4326'})

        inv.geometry.plot(ax=axs['map'],color='white', edgecolor=None,zorder=3)



        ne.geometry.boundary.plot(ax=axs['map'],color=None,edgecolor='#616161',linewidth=1,zorder=4)

        ## scatter installations
        df['x_clip'] = np.round(df['x']/self.params['agg'],0)*self.params['agg']
        df['y_clip'] = np.round(df['y']/self.params['agg'],0)*self.params['agg']

        df['xy'] = list(zip(df['x_clip'], df['y_clip']))
        df['dt_obj_days'] = (df['dt_obj'] - pd.to_datetime('2016-06-01')).dt.days

        # get the MW capacity
        df_agg = df[['xy','capacity_mw']].groupby('xy').sum() # that gets all the xys


        # get the install date
        df_agg['dt_obj'] = df.loc[df['dt_obj_days']>0,['xy','dt_obj_days']].groupby('xy').mean()

        print ('dodo')
        print (df.loc[df['xy'].isin(df_agg[df_agg['dt_obj'].isna()].index.values),:])
        print (df.loc[df['xy'].isin(df_agg[df_agg['dt_obj'].isna()].index.values),['xy','dt_obj_days']].groupby('xy').mean())

        df_agg.loc[df_agg['dt_obj'].isna(), 'dt_obj'] = df.loc[df['xy'].isin(df_agg[df_agg['dt_obj'].isna()].index.values),['xy','dt_obj_days']].groupby('xy').mean()['dt_obj_days']

        print (df_agg)

        ss = np.array(
            [(self.params['minsize']+((self.params['maxsize']-self.params['minsize'])*((np.log10(mw)+1)-self.params['minmw'])/(self.params['maxmw']-self.params['minmw']))) 
            for mw in df_agg['capacity_mw'].values])


        colmap = cm.get_cmap('cool_r',256)

        cmaplist = [colmap(i) for i in range(colmap.N)]
        # force the first color entry to be grey
        cmaplist = [(133/255, 0, 119/255,1)]*3 + cmaplist

        # create the new map
        cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, colmap.N)

        # define the bins and normalize
        bounds = np.linspace(0, 7, 8)
        norm = colors.BoundaryNorm(bounds, colmap.N)

        print ('df agg',df_agg)

        date_delta_max = (dt.strptime('2018-12-31','%Y-%m-%d') - dt.strptime('2016-06-01','%Y-%m-%d')).days

        cs = np.array([(0.2 + 0.8*(max(date_delta/date_delta_max,0)))*256 for date_delta in df_agg['dt_obj']])
        cs = np.array(colmap(np.nan_to_num(cs).astype(int)))

        axs['map'].scatter(
            x = df_agg[~df_agg['dt_obj'].isna()].index.str[0],
            y = df_agg[~df_agg['dt_obj'].isna()].index.str[1],
            marker="o",
            s=ss[~df_agg['dt_obj'].isna()],
            c=cs[~df_agg['dt_obj'].isna()],
            #edgecolors='grey',
            alpha=0.5,
            zorder=5)

        axs['map'].scatter(
            x = df_agg[df_agg['dt_obj'].isna()].index.str[0],
            y = df_agg[df_agg['dt_obj'].isna()].index.str[1],
            marker="o",
            s=ss[df_agg['dt_obj'].isna()],
            c=(133/255, 0, 119/255,1),
            #edgecolors='grey',
            alpha=0.5,
            zorder=5)




        axs['map'].set_ylim([-60,85])
        axs['map'].set(zorder=6)


        
        #axs['scatter'].set_xticks([np.datetime64(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)).astype(np.int64)*1000 for mm in range(0,33,3)])
        
        ### base
        records = {}
        # World
        rec ={'nd':df.loc[df['dt_obj'].isna(),'capacity_mw'].sum(), '<2016-06':df.loc[df['dt_obj']<dt.strptime('2016-06-30','%Y-%m-%d'),'capacity_mw'].sum()}

        for mm in range(5):
            dt_indices = (df['dt_obj']>=(dt.strptime('2016-06-30','%Y-%m-%d')+relativedelta(months=mm*6))) & (df['dt_obj']<(dt.strptime('2016-06-30','%Y-%m-%d')+relativedelta(months=(mm+1)*6)))
            rec[(dt.strptime('2016-06-30','%Y-%m-%d')+relativedelta(months=(mm+1)*6)).isoformat()[0:10]] = df.loc[dt_indices,'capacity_mw'].sum()

        records['WORLD']=rec

        # CN
        rec ={'nd':df.loc[(df['dt_obj'].isna()) & (df['iso-3166-1']=='CN'),'capacity_mw'].sum(), '<2016-06':df.loc[(df['dt_obj']<dt.strptime('2016-06-30','%Y-%m-%d')) & (df['iso-3166-1']=='CN'),'capacity_mw'].sum()}

        for mm in range(5):
            dt_indices = (df['dt_obj']>=(dt.strptime('2016-06-30','%Y-%m-%d')+relativedelta(months=mm*6))) & (df['dt_obj']<(dt.strptime('2016-06-30','%Y-%m-%d')+relativedelta(months=(mm+1)*6))) & (df['iso-3166-1']=='CN')
            rec[(dt.strptime('2016-06-30','%Y-%m-%d')+relativedelta(months=(mm+1)*6)).isoformat()[0:10]] = df.loc[dt_indices,'capacity_mw'].sum()

        records['CN']=rec

        # EU+GB
        EU_28 = ['AT','BE','BG','CY','CZ','DK','EE','FI','FR','DE','GR','HU','IE','IT','LV','LT','LU','MT','NL','PL','PT','RO','SK','SI','ES','SE','GB']
        rec ={'nd':df.loc[(df['dt_obj'].isna()) & (df['iso-3166-1'].isin(EU_28)),'capacity_mw'].sum(), '<2016-06':df.loc[(df['dt_obj']<dt.strptime('2016-06-30','%Y-%m-%d')) & (df['iso-3166-1'].isin(EU_28)),'capacity_mw'].sum()}

        for mm in range(5):
            dt_indices = (df['dt_obj']>=(dt.strptime('2016-06-30','%Y-%m-%d')+relativedelta(months=mm*6))) & (df['dt_obj']<(dt.strptime('2016-06-30','%Y-%m-%d')+relativedelta(months=(mm+1)*6))) & (df['iso-3166-1'].isin(EU_28))
            rec[(dt.strptime('2016-06-30','%Y-%m-%d')+relativedelta(months=(mm+1)*6)).isoformat()[0:10]] = df.loc[dt_indices,'capacity_mw'].sum()

        records['EU+GB']=rec

        for isostr in ['US','IN','JP','DE','IT','ES','GB','TR','FR','CL','ZA','MX','TH','AU','KR','CA','CZ','GR','RO']:
            rec ={'nd':df.loc[(df['dt_obj'].isna()) & (df['iso-3166-1']==isostr),'capacity_mw'].sum(), '<2016-06':df.loc[(df['dt_obj']<dt.strptime('2016-06-30','%Y-%m-%d')) & (df['iso-3166-1']==isostr),'capacity_mw'].sum()}

            for mm in range(5):
                dt_indices = (df['dt_obj']>=(dt.strptime('2016-06-30','%Y-%m-%d')+relativedelta(months=mm*6))) & (df['dt_obj']<(dt.strptime('2016-06-30','%Y-%m-%d')+relativedelta(months=(mm+1)*6))) & (df['iso-3166-1']==isostr)
                rec[(dt.strptime('2016-06-30','%Y-%m-%d')+relativedelta(months=(mm+1)*6)).isoformat()[0:10]] = df.loc[dt_indices,'capacity_mw'].sum()

            records[isostr]=rec

        df_bar = pd.DataFrame.from_dict(records)

        print ('df bar',df_bar)

        

        bar_cols = [(133/255, 0, 119/255,1)] + [colmap(int((0.2+0.8*(ii/5))*255)) for ii in range(6)]

        print(bar_cols)

        df_bar = df_bar.T/1000

        df_bar.loc[['WORLD','CN'],:].plot.bar(ax=axs['capmw_0'], stacked=True, color=bar_cols, legend=False)
        df_bar.loc[['EU+GB','US','IN','JP','DE','IT','ES','GB'],:].plot.bar(ax=axs['capmw_1'], stacked=True, color=bar_cols, legend=False)
        df_bar.loc[['TR','FR','CL','ZA','MX','TH','AU','KR','CA','CZ','GR','RO'],:].plot.bar(ax=axs['capmw_2'], stacked=True, color=bar_cols, legend=False)
        axs['capmw_0'].set_ylabel('Est. Generating Capacity $[GW]$')
        #axs['capmw'].set_yscale('log')




        """  timeseries
        df.set_index('dt_obj')['capacity_mw'].cumsum().plot(ax=axs['capmw'],c='k')
        EU_28 = ['AT','BE','BG','CY','CZ','DK','EE','FI','FR','DE','GR','HU','IE','IT','LV','LT','LU','MT','NL','PL','PT','RO','SK','SI','ES','SE','GB']
        axs['capmw'].text(dt.strptime('2018-10-31','%Y-%m-%d'),df['capacity_mw'].sum(),'WORLD', verticalalignment='center')
        for isostr in ['CN','US','IN','JP']:
            df[df['iso-3166-1']==isostr].set_index('dt_obj')['capacity_mw'].cumsum().plot(ax=axs['capmw'],c='k')
            if isostr=='US':
                axs['capmw'].text(dt.strptime('2018-10-31','%Y-%m-%d'),df.loc[df['iso-3166-1']==isostr,'capacity_mw'].sum(),isostr, verticalalignment='top')
            else:
                axs['capmw'].text(dt.strptime('2018-10-31','%Y-%m-%d'),df.loc[df['iso-3166-1']==isostr,'capacity_mw'].sum(),isostr, verticalalignment='center')

        df[df['iso-3166-1'].isin(EU_28)].set_index('dt_obj')['capacity_mw'].cumsum().plot(ax=axs['capmw'],c='k')
        axs['capmw'].text(dt.strptime('2018-10-31','%Y-%m-%d'),df.loc[df['iso-3166-1'].isin(EU_28),'capacity_mw'].sum(),'EU+GB')

        axs['capmw'].set_xticks([(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)) for mm in range(0,33,3)])
        axs['capmw'].set_xlabel('')
        axs['capmw'].set_yscale('log')
        #axs['capmw'].set_ylim([10,None])
        axs['capmw'].set_xlim(['2016-06-01','2018-10-15'])
        axs['capmw'].set_ylabel('Est. Generating Capacity $[GW]$')
        axs['capmw'].set_xticklabels([str((dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)))[0:7] for mm in range(0,33,3)])
        """

        axs_inset = fig.add_axes([axs['legend'].get_position().x0,axs['legend'].get_position().y0,0.02,axs['legend'].get_position().height])

        cb = ColorbarBase(axs_inset, cmap=cmap, norm=norm, spacing='proportional', ticks=[0.5+ii for ii in range(7)], boundaries=bounds, format='%1i', orientation='vertical')
        
        cb.ax.set_yticklabels(['No Date','<2016-06-30','.. - 2016-12-30','.. - 2017-06-30','.. - 2017-12-31','.. - 2018-06-30','.. - 2018-09-31'])
        #leg_keys = ['Forest','Grassland','Wetland','Cropland','Developed','Barren/Other']
            
        #custom_lines = []
        #for kk,vv in self.colors_dict.items():
        #    custom_lines.append(Line2D([0],[0],color=vv,marker='.',linestyle=None, lw=0, markersize=10))

        for kk in ['capmw_0', 'capmw_1','capmw_2']:
            axs[kk].spines['right'].set_visible(False)
            axs[kk].spines['top'].set_visible(False)


        axs['legend'].axis('off')
        #axs['legend'].legend(custom_lines, leg_keys, ncol=1, loc='center', handletextpad=0.1, columnspacing=0.1, fontsize=18, frameon=False)

        plt.savefig(os.path.join(os.getcwd(),'makefigs','fig2-global.png'))#, bbox_extra_artists=(lgd,))
        plt.show()






    def make_arr(self):

        arr = np.zeros((360*10,180*10,6))

        ind_dict = dict(zip(['cropland', 'forestshrub', 'grassy', 'human', 'other', 'wetlands'],range(6)))

        for ii_r,row in enumerate(self.df.iterrows()):
            if ii_r % 1000 ==0:
                print ('ii_r')
                print (row)
            x = int((row[1]['x'] + 180)*10)
            y = int((row[1]['y'] + 90)*10)

            if row[1]['land_cover_vis'] in ['cropland', 'forestshrub', 'grassy', 'human', 'other', 'wetlands']:

                arr[x,y,ind_dict[row[1]['land_cover_vis']]] += row[1]['area']

        np.savez(os.path.join(os.getcwd(),'data','land_cover_arr.npz'),arr=arr)





if __name__=="__main__":
    generator=Figure(make_arr=True)
    #generator.make_arr()
    generator.make_global()
    #generator.make_regional()
