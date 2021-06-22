import pickle, logging, os, sys, json
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

import pandas as pd
import geopandas as gpd

import numpy as np
from shapely import geometry
from skimage.measure import block_reduce

gpd.options.use_pygeos=False

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from matplotlib import colors

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class Figure:

    def __init__(self):

        logger.info(f'initialising...')

        self.classes = ['cropland','urban_areas','treecover','grasslands','shrub/herbaceous/sparse','desert','barren_areas','wetlands','other']
        self.leg_keys = ['Cropland','Built-up Areas','Forest','Grassland','Shrubland','Aridlands','Barren Land','Wetlands','Other']
        

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

        self.class_map = json.load(open(os.path.join(os.getcwd(),'data','lc300_classes.json'),'r'))

        logger.info(f'loading features...')

        gdf = gpd.read_file(os.path.join(os.getcwd(),'data','SPV_v5.gpkg'))


        logger.info(f'mapping dates...')

        gdf['install_date'] = gdf['install_date'].str.replace('<2016-06','2000-01-01')
        gdf['install_date'] = gdf['install_date'].str.replace(',','')
        gdf['install_date'] = gdf['install_date'].str[0:10]

        gdf['dt_obj'] = pd.to_datetime(gdf['install_date'])



        df = pd.DataFrame(gdf[['capacity_mw','area','iso-3166-1','dt_obj','lc_vis']])

        logger.info(f'Cast to dummies')
        df = df.merge(pd.get_dummies(df['lc_vis']), how='left', left_index=True, right_index=True)
        self.df = df[df.area>10^1]

        for kk in self.classes:
            self.df[kk] = self.df[kk]*self.df['capacity_mw']/1000


        self.df_lcpix = pd.read_csv(os.path.join(os.getcwd(),'data','LC300_latlonpixiso2.csv')).set_index(['Unnamed: 0','Unnamed: 1'])
        
        self.df_lcworld =  pd.read_csv(os.path.join(os.getcwd(),'data','LC300_world.csv')).reset_index().rename(columns={'Unnamed: 0':'ISO_A2'})

        for kk,vv in self.class_map.items():
            self.df_lcpix[kk] = self.df_lcpix.loc[:,vv].astype('float').sum(axis=1)
            self.df_lcworld[kk] = self.df_lcworld.loc[:,vv].astype('float').sum(axis=1)


        self.df = self.df.reset_index().sort_values('dt_obj', na_position='first').set_index('index')

        self.df = pd.merge(self.df, self.df[self.classes].cumsum()/1000, how='left',left_index=True, right_index=True, suffixes=('','_cumsum'))


        self.df = self.df.set_index('dt_obj')

        logger.info(f'Instantiated.')


    def make_fig(self):

        fig = plt.figure(figsize=(12,8))

        gs = GridSpec(3, 4, figure=fig, wspace=0.3, hspace=0.2)
        gs0 = GridSpec(1,1, figure=fig, wspace=0.3, hspace=0.2)

        axs = {}

        axs['legend'] = fig.add_subplot(gs0[0,0])
        axs['legend'].axis('off')

        for ii in range(3):
            axs[ii] = {}
            axs[ii]['ts'] = fig.add_subplot(gs[ii,0:2])
            axs[ii]['local_skew'] = fig.add_subplot(gs[ii,2])
            axs[ii]['global_skew'] = fig.add_subplot(gs[ii,3])

        for ii, bin_bounds in enumerate([(0,1),(1,5),(5,4000)]):
            logger.info(f'{ii}, {bin_bounds}')

            df_lcpv =  pd.read_csv(os.path.join(os.getcwd(),'data',f'LC300_pvpix_{bin_bounds[0]}.csv')).set_index(['Unnamed: 0','Unnamed: 1'])
            df_lcpv = df_lcpv[[cc for cc in df_lcpv if cc not in ['index','iso2']]].astype(float)

            df_lcpix = self.df_lcpix.loc[self.df_lcpix.index.isin(df_lcpv.index.values),:]

            tot_pix = df_lcpix[self.classes].sum().sum()
            tot_pv = df_lcpv[self.classes].sum().sum()
            tot_world = self.df_lcworld[self.classes].sum().sum()

            self.df.loc[(self.df['capacity_mw']>=bin_bounds[0]) & (self.df['capacity_mw']<bin_bounds[1]),self.classes].cumsum().clip(0).plot.area(ax=axs[ii]['ts'], color=[self.colors_dict[kk] for kk in self.classes],legend=False, lw=0)
            self.df.loc[(self.df['capacity_mw']>=bin_bounds[0]) & (self.df['capacity_mw']<bin_bounds[1]),self.classes].cumsum().clip(0).to_csv(os.path.join(os.getcwd(),'makefigs','data',f'fig-A11-{ii}-ts.csv'))


            diff = df_lcpv[self.classes].sum()/tot_pv - df_lcpix[self.classes].sum()/tot_pix 
            diff.to_csv(os.path.join(os.getcwd(),'makefigs','data',f'fig-A11-{ii}-localskew.csv'))

            diff.plot.barh(ax=axs[ii]['local_skew'], color=[self.colors_dict[kk] for kk in self.classes])

            # get only pix that are in that bin csv
            diff =  df_lcpix[self.classes].sum()/tot_pix - self.df_lcworld[self.classes].sum()/tot_world
            diff.to_csv(os.path.join(os.getcwd(),'makefigs','data',f'fig-A11-{ii}-globalskew.csv'))

            diff.plot.barh(ax=axs[ii]['global_skew'], color=[self.colors_dict[kk] for kk in self.classes])

            axs[ii]['ts'].set_xticks([(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)) for mm in range(0,33,3)])
            axs[ii]['ts'].set_xlabel('')
            axs[ii]['ts'].set_xlim(['2016-06-01','2018-10-15'])
            axs[ii]['ts'].set_xticklabels([])


            axs[ii]['local_skew'].axvline(0, color='k', lw=0.5)
            axs[ii]['local_skew'].spines['right'].set_visible(False)
            axs[ii]['local_skew'].spines['top'].set_visible(False)
            axs[ii]['local_skew'].spines['left'].set_visible(False)
            axs[ii]['local_skew'].set_yticks([])
            axs[ii]['local_skew'].set_yticklabels([])
            axs[ii]['local_skew'].set_xlim([-.2,.2])
            axs[ii]['local_skew'].set_xticks([-.2,0,.2])
            axs[ii]['local_skew'].set_xticklabels(['-20%','0%','20%'])

            axs[ii]['global_skew'].axvline(0, color='k', lw=0.5)
            axs[ii]['global_skew'].spines['right'].set_visible(False)
            axs[ii]['global_skew'].spines['top'].set_visible(False)
            axs[ii]['global_skew'].spines['left'].set_visible(False)
            axs[ii]['global_skew'].set_yticks([])
            axs[ii]['global_skew'].set_yticklabels([])
            axs[ii]['global_skew'].set_xlim([-.5,.5])
            axs[ii]['global_skew'].set_xticks([-0.5, 0, 0.5])
            axs[ii]['global_skew'].set_xticklabels(['-50%','0%','50%'])

        axs[2]['ts'].set_xticklabels([str((dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)))[0:7] for mm in range(0,33,3)])
        axs[0]['ts'].set_title('(a)')
        axs[0]['local_skew'].set_title('(b)')
        axs[0]['global_skew'].set_title('(c)')

        plt.annotate('Est. Generating Capacity $[GW]$',horizontalalignment='left', xy=(45,800-500), xycoords='figure pixels', fontsize=14, rotation=90) #fontweight='bold'
        #axs[ii]['ts'].set_ylabel()
        axs[0]['ts'].set_ylabel('10kW - 1MW (N=38,288)')
        axs[1]['ts'].set_ylabel('1MW - 5MW (N=20,801)')
        axs[2]['ts'].set_ylabel('>5MW (N=9,572)')

        leg_keys = self.leg_keys

            
        custom_lines = []
        for kk in self.classes:
            custom_lines.append(Line2D([0],[0],color=self.colors_dict[kk],marker='.',linestyle=None, lw=0, markersize=13))

        axs['legend'].legend(custom_lines, leg_keys, ncol=len(self.classes), loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=False, shadow=False, frameon=False)

        #axs['legend'].axis('off')
        #axs['legend'].legend(custom_lines, list(self.colors_dict.keys()), ncol=2, loc='center', handletextpad=0.1, fontsize=18)
         
        plt.annotate('Anthromes',horizontalalignment='left', xy=(53, 800-710), xycoords='figure pixels', fontsize=13) #fontweight='bold'
        plt.annotate('Biomes',horizontalalignment='left', xy=(338, 800-710), xycoords='figure pixels', fontsize=13)

        plt.annotate("",
            xy=(338,800-715), xycoords='figure pixels',
            xytext=(1185,800-715), textcoords='figure pixels',
            arrowprops=dict(width=0.3, headwidth=0.3,facecolor='k',headlength=0.001,shrink=0.001),
            )

        plt.annotate("",
            xy=(53,800-715), xycoords='figure pixels',
            xytext=(300,800-715), textcoords='figure pixels',
            arrowprops=dict(width=0.3, headwidth=0.3,facecolor='k',headlength=0.001,shrink=0.001),
            )
        


        plt.savefig(os.path.join(os.getcwd(),'makefigs','figures','fig-A11_land_cover_binned.png'))
        plt.show()

if __name__ == "__main__":

    generator = Figure()
    generator.make_fig()