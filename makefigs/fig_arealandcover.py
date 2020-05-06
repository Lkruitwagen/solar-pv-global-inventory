import pickle, logging, os, sys, json
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logger = logging.getLogger(__name__)


class Figure:

    def __init__(self):

        logger.info(f'initialising...')

        self.colors_dict = {
            'forest':'#2f9149',
            'grasslands':'#b6eb7f',
            'wetlands':'#7fe7eb',
            'cropland':'#ff9500',
            'developed':'#d000ff',
            'other':'#bdbdbd',
        }

        self.colors_base = {
            'forestshrub':'#2f9149',
            'grassy':'#b6eb7f',
            'wetlands':'#7fe7eb',
            'cropland':'#ff9500',
            'human':'#d000ff',
            'other':'#bdbdbd',
            np.nan:'#bdbdbd'
        }

        self.corine_countries = ['AL', 'AT', 'BE', 'BA', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IS', 'IE', 'IT', 'XK', 'LV', 'LI', 'LT', 'LU', 'MK', 'MT', 'ME', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SK', 'SI', 'ES', 'SE', 'CH', 'TR', 'GB']

        logger.info(f'loading features...')

        landcover_fts = json.load(open(os.path.join(os.getcwd(), 'data','ABCD_landcover.geojson'),'r'))['features']

        df = pd.DataFrame.from_records([ft['properties'] for ft in landcover_fts])

        logger.info(f'mapping landcover to classes...')

        df['install_date'] = df['install_date'].str.replace('<2016-06','')
        df['install_date'] = df['install_date'].str.replace(',','')
        df['install_date'] = df['install_date'].str[0:10]
        df['dt_obj'] = pd.to_datetime(df['install_date'])

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

        df_cdl = pd.DataFrame.from_dict(labels_agg['CDL'], orient='index')
        df_cdl = df_cdl.unstack().dropna().reset_index().set_index(0).drop(columns=['level_0']).sort_index()
        df_cdl.index = df_cdl.index.astype(int)

        df_corine = pd.DataFrame.from_dict(labels_agg['CORINE'], orient='index')
        df_corine = df_corine.unstack().dropna().reset_index().set_index(0).drop(columns=['level_0']).sort_index()
        df_corine.index = df_corine.index.astype(int)

        df_modis = pd.DataFrame.from_dict(labels_agg['MODIS'], orient='index')
        df_modis = df_modis.unstack().dropna().reset_index().set_index(0).drop(columns=['level_0']).sort_index()
        df_modis.index = df_modis.index.astype(int)

        df['land_cover_vis'] = ''
        df.loc[df['iso-3166-1']=='US','land_cover_vis'] = df.loc[df['iso-3166-1']=='US','land_cover_CDL_2012'].map(df_cdl.to_dict()['level_1'])
        df.loc[df['iso-3166-1'].isin(self.corine_countries),'land_cover_vis'] = df.loc[df['iso-3166-1'].isin(self.corine_countries),'land_cover_CORINE_2012'].map(df_corine.to_dict()['level_1'])
        df.loc[~df['iso-3166-1'].isin(self.corine_countries+['US']),'land_cover_vis'] = df.loc[~df['iso-3166-1'].isin(self.corine_countries+['US']),'land_cover_MODIS_2012'].map(df_modis.to_dict()['level_1'])

        df = df[['area','iso-3166-1','dt_obj','land_cover_vis']]
        df = df.merge(pd.get_dummies(df['land_cover_vis']), how='left', left_index=True, right_index=True)
        self.df = df[df.area>10^1]

    def make_global(self):
        
        df_ndt = self.df[self.df['dt_obj'].isna()]
        df = self.df[~self.df['dt_obj'].isna()].sort_values('dt_obj')

        print (df)


        for cc in ['cropland', 'forestshrub', 'grassy', 'human', 'other', 'wetlands']:
            df[cc+'_orig'] = df[cc]
            df[cc] = df[cc]*df['area']

            df_ndt[cc+'_orig'] = df_ndt[cc]
            df_ndt[cc] = df_ndt[cc]*df_ndt['area']

        df = df.merge(df[['dt_obj','cropland','forestshrub','grassy','human','other','wetlands']].rolling('91D', on='dt_obj').sum(), how='left',left_index=True, right_index=True, suffixes=('','_rolling'))

        df['rolling_sum'] = df[['cropland_rolling','forestshrub_rolling','grassy_rolling','human_rolling','other_rolling','wetlands_rolling']].sum(axis=1)    

        for cc in ['cropland', 'forestshrub', 'grassy', 'human', 'other', 'wetlands']:
            df[cc] = df[cc+'_rolling']/df['rolling_sum']

        df = df.rename(columns={'forestshrub':'forest', 'grassy':'grasslands', 'human':'developed'})

        df['ts'] = df['dt_obj'].astype(np.int64)

        df['colors'] = df['land_cover_vis'].apply(lambda x: self.colors_base[x])


        fig = plt.figure(figsize=(30,20))

        gs = GridSpec(4, 6, figure=fig, wspace=0.2, hspace=0.2)

        axs = {}

        axs['scatter'] = fig.add_subplot(gs[0:3,0:5])
        axs['area'] = fig.add_subplot(gs[3,0:5])
        axs['dist'] = fig.add_subplot(gs[0:3,5])
        axs['legend'] = fig.add_subplot(gs[3,5])



        df[['ts','area','cropland', 'forest', 'grasslands', 'developed', 'other', 'wetlands']].plot.scatter(ax=axs['scatter'],x='ts',y='area', color=df['colors'].values,logy=True, alpha=0.3)
        df.set_index('dt_obj')[['cropland', 'forest', 'grasslands', 'developed', 'other', 'wetlands']].clip(0).plot.area(ax=axs['area'], color=[self.colors_dict[kk] for kk in ['cropland', 'forest', 'grasslands', 'developed', 'other', 'wetlands']],legend=False)

        axs['scatter'].set_xticks([np.datetime64(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)).astype(np.int64)*1000 for mm in range(0,33,3)])
        axs['area'].set_xticks([(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)) for mm in range(0,33,3)])
        
        axs['scatter'].set_xticklabels(['' for _ in axs['scatter'].get_xticks()])
        axs['area'].set_xticklabels(['' for mm in range(0,33,3)])
        #else:
        #    axs['area'].set_xticklabels([str((dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)))[0:7] for mm in range(0,33,3)])


        for col in ['cropland','forestshrub','grassy','human','other']:
            np.log10(df[df[col+'_orig']>0].area).hist(ax=axs['dist'], bins=10, alpha=0.75, edgecolor=self.colors_base[col],histtype='step', linewidth=3,density=True, orientation='horizontal', fill=False)

        for col in ['cropland','forestshrub','grassy','human','other']:
            np.log10(df_ndt[df_ndt[col+'_orig']>0].area).hist(ax=axs['dist'], bins=10, alpha=0.75, edgecolor=self.colors_base[col],histtype='step', linestyle='--',linewidth=2,density=True, orientation='horizontal', fill=False)

        

        axs['scatter'].set_ylabel('Area [$m^2$]')
        axs['area'].set_xlabel('')
        axs['scatter'].set_xlabel('')
        
        axs['area'].set_xlim(['2016-06-01','2018-10-31'])
        axs['scatter'].set_xlim([np.datetime64(dt.strptime('2016-06-01','%Y-%m-%d')).astype(np.int64)*1000, np.datetime64(dt.strptime('2018-10-31','%Y-%m-%d')).astype(np.int64)*1000])
        axs['scatter'].set_ylim([10**2, 10**7])
        axs['dist'].set_ylim([2, 7])
        axs['dist'].set_yticklabels([])
        axs['dist'].grid(False)
        axs['dist'].set_xticklabels([])

        axs['area'].yaxis.set_major_formatter(mtick.PercentFormatter(1))
        #axs['scatter'].text(np.datetime64(dt.strptime('2018-09-01','%Y-%m-%d')).astype(np.int64)*1000,10**6.5,kk, fontsize=20)
        
        #axs[kk]['area'] = fig.add_subplot(gs[(ii_k%2)*4+3,ii_k//2])
        
        
        #print ([ts for ts in ax.get_xticks()])
        axs['area'].set_xticklabels([str((dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)))[0:7] for mm in range(0,33,3)])


            
        custom_lines = []
        for kk,vv in self.colors_dict.items():
            custom_lines.append(Line2D([0],[0],color=vv,marker='.',linestyle=None, lw=0))
        custom_lines.append(Line2D([0],[0],color='gray',marker=None,linestyle='-', lw=2))
        custom_lines.append(Line2D([0],[0],color='gray',marker=None,linestyle='--', lw=1))


        axs['legend'].axis('off')
        axs['legend'].legend(custom_lines, list(self.colors_dict.keys())+['dated','not dated'], ncol=2, loc='center')

        plt.savefig('./land_cover_global.png')#, bbox_extra_artists=(lgd,))
        plt.show()





    def make_regional(self):

        dfs = {}
        dfs_ndt = {}

        dfs['US'] = self.df.loc[self.df['iso-3166-1']=='US',:].sort_values('dt_obj')
        dfs['CN'] = self.df.loc[self.df['iso-3166-1']=='CN',:].sort_values('dt_obj')
        dfs['IN'] = self.df.loc[self.df['iso-3166-1']=='IN',:].sort_values('dt_obj')
        dfs['EU'] = self.df.loc[self.df['iso-3166-1'].isin(self.corine_countries),:].sort_values('dt_obj')

        for kk in dfs.keys():
            dfs_ndt[kk] = df[df['dt_obj'].isna()]
            df = df[~df['dt_obj'].isna()]

        for kk in dfs.keys():
            df = df.merge(df[['dt_obj','cropland','forestshrub','grassy','human','other','wetlands']].rolling('91D', on='dt_obj').sum(), how='left',left_index=True, right_index=True, suffixes=('','_rolling'))

            df['rolling_sum'] = df[['cropland_rolling','forestshrub_rolling','grassy_rolling','human_rolling','other_rolling','wetlands_rolling']].sum(axis=1)    

            for cc in ['cropland', 'forestshrub', 'grassy', 'human', 'other', 'wetlands']:
                df[cc] = df[cc+'_rolling']/df['rolling_sum']

            df = df.rename(columns={'forestshrub':'forest', 'grassy':'grasslands', 'human':'developed'})

            df['ts'] = df['dt_obj'].astype(np.int64)

            df['colors'] = df['land_cover_vis'].apply(lambda x: self.colors_base[x])


        fig = plt.figure(figsize=(30,20))

        gs = GridSpec(8, 2, figure=fig)

        axs = {kk:{} for kk in ['US','CN','IN','EU']}

        for ii_k, kk in enumerate(['US','CN','IN','EU']):
            axs[kk]['scatter'] = fig.add_subplot(gs[(ii_k%2)*4:(ii_k%2)*4+3,ii_k//2])
            axs[kk]['area'] = fig.add_subplot(gs[(ii_k%2)*4+3,ii_k//2])


        for kk in ['US','CN','IN','EU']:
            dfs[kk][['ts','area','cropland', 'forest', 'grasslands', 'developed', 'other', 'wetlands']].plot.scatter(ax=axs[kk]['scatter'],x='ts',y='area', color=dfs[kk]['colors'].values,logy=True, alpha=0.3)
            dfs[kk].set_index('dt_obj')[['cropland', 'forest', 'grasslands', 'developed', 'other', 'wetlands']].plot.area(ax=axs[kk]['area'], color=[self.colors_dict[kk] for kk in ['cropland', 'forest', 'grasslands', 'developed', 'other', 'wetlands']],legend=False)

            
        for ii_k, kk in enumerate(['US','CN','IN','EU']):
            axs[kk]['scatter'].set_xticks([np.datetime64(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)).astype(np.int64)*1000 for mm in range(0,33,3)])
            axs[kk]['area'].set_xticks([(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)) for mm in range(0,33,3)])
            
            axs[kk]['scatter'].set_xticklabels(['' for _ in axs[kk]['scatter'].get_xticks()])
            if kk in ['US','IN']:
                axs[kk]['area'].set_xticklabels(['' for mm in range(0,33,3)])
            else:
                axs[kk]['area'].set_xticklabels([str((dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)))[0:7] for mm in range(0,33,3)])
            #print ( [pd.Timestamp(tt*1000*10) for tt in axs[kk]['area'].get_xticks()])
            #axs[kk]['area'].set_xticklabels([ axs[kk]['area'].get_xticks()])
            

            axs[kk]['scatter'].set_ylabel('Area [$m^2$]')
            axs[kk]['area'].set_xlabel('')
            axs[kk]['scatter'].set_xlabel('')
            
            axs[kk]['area'].set_xlim(['2016-06-01','2018-10-31'])
            axs[kk]['scatter'].set_xlim([np.datetime64(dt.strptime('2016-06-01','%Y-%m-%d')).astype(np.int64)*1000, np.datetime64(dt.strptime('2018-10-31','%Y-%m-%d')).astype(np.int64)*1000])
            axs[kk]['scatter'].set_ylim([10**2, 10**7])
            axs[kk]['area'].yaxis.set_major_formatter(mtick.PercentFormatter(1))
            axs[kk]['scatter'].text(np.datetime64(dt.strptime('2018-09-01','%Y-%m-%d')).astype(np.int64)*1000,10**6.5,kk, fontsize=20)
            
            #axs[kk]['area'] = fig.add_subplot(gs[(ii_k%2)*4+3,ii_k//2])
            
            
            #print ([ts for ts in ax.get_xticks()])
            #axs[kk]['scatter'].set_xticklabels([dt.fromtimestamp(ts).strftime('%Y-%m') for ts in ax.get_xticks()])
            
        custom_lines = []
        for kk,vv in self.colors_dict.items():
            custom_lines.append(Line2D([0],[0],color=vv,marker='.',linestyle=None, lw=0))


        lgd = fig.legend(custom_lines, self.colors_dict.keys(), ncol=6, bbox_to_anchor=(0.5, 0.15), loc='center',fancybox=True)

        plt.savefig('./land_cover.png', bbox_extra_artists=(lgd,))
        plt.show()

    def make_regional_area_dist(self):

        dfs = {}
        dfs_ndt = {}

        dfs['US'] = self.df.loc[self.df['iso-3166-1']=='US',:].sort_values('dt_obj')
        dfs['CN'] = self.df.loc[self.df['iso-3166-1']=='CN',:].sort_values('dt_obj')
        dfs['IN'] = self.df.loc[self.df['iso-3166-1']=='IN',:].sort_values('dt_obj')
        dfs['EU'] = self.df.loc[self.df['iso-3166-1'].isin(self.corine_countries),:].sort_values('dt_obj')

        for kk in dfs.keys():
            dfs_ndt[kk] = dfs[kk][dfs[kk]['dt_obj'].isna()]
            dfs[kk] = dfs[kk][~dfs[kk]['dt_obj'].isna()]


        for kk in dfs.keys():
            for cc in ['cropland', 'forestshrub', 'grassy', 'human', 'other', 'wetlands']:
                dfs[kk][cc+'_orig'] = dfs[kk][cc]
                dfs[kk][cc] = dfs[kk][cc]*dfs[kk]['area']

        for kk in dfs.keys():
            dfs[kk] = dfs[kk].merge(dfs[kk][['dt_obj','cropland','forestshrub','grassy','human','other','wetlands']].rolling('91D', on='dt_obj').sum(), how='left',left_index=True, right_index=True, suffixes=('','_rolling'))

            dfs[kk]['rolling_sum'] = dfs[kk][['cropland_rolling','forestshrub_rolling','grassy_rolling','human_rolling','other_rolling','wetlands_rolling']].sum(axis=1)    

            for cc in ['cropland', 'forestshrub', 'grassy', 'human', 'other', 'wetlands']:
                dfs[kk][cc] = dfs[kk][cc+'_rolling']/dfs[kk]['rolling_sum']

            dfs[kk] = dfs[kk].rename(columns={'forestshrub':'forest', 'grassy':'grasslands', 'human':'developed'})

            dfs[kk]['ts'] = dfs[kk]['dt_obj'].astype(np.int64)

            dfs[kk]['colors'] = dfs[kk]['land_cover_vis'].apply(lambda x: self.colors_base[x])


        fig = plt.figure(figsize=(30,20))

        gs = GridSpec(8, 12, figure=fig, wspace=0.2, hspace=0.2)

        axs = {kk:{} for kk in ['US','CN','IN','EU']}

        for ii_k, kk in enumerate(['US','CN','IN','EU']):
            axs[kk]['scatter'] = fig.add_subplot(gs[(ii_k%2)*4:(ii_k%2)*4+3,(ii_k//2)*6:(ii_k//2)*6+5])
            axs[kk]['area'] = fig.add_subplot(gs[(ii_k%2)*4+3,(ii_k//2)*6:(ii_k//2)*6+5])
            axs[kk]['dist'] = fig.add_subplot(gs[(ii_k%2)*4:(ii_k%2)*4+3,  (ii_k//2)*6+5 ])


        for kk in ['US','CN','IN','EU']:
            dfs[kk][['ts','area','cropland', 'forest', 'grasslands', 'developed', 'other', 'wetlands']].plot.scatter(ax=axs[kk]['scatter'],x='ts',y='area', color=dfs[kk]['colors'].values,logy=True, alpha=0.3)
            dfs[kk].set_index('dt_obj')[['cropland', 'forest', 'grasslands', 'developed', 'other', 'wetlands']].clip(0).plot.area(ax=axs[kk]['area'], color=[self.colors_dict[kk] for kk in ['cropland', 'forest', 'grasslands', 'developed', 'other', 'wetlands']],legend=False)

            axs[kk]['scatter'].set_xticks([np.datetime64(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)).astype(np.int64)*1000 for mm in range(0,33,3)])
            axs[kk]['area'].set_xticks([(dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)) for mm in range(0,33,3)])
            
            axs[kk]['scatter'].set_xticklabels(['' for _ in axs[kk]['scatter'].get_xticks()])
            if kk in ['US','IN']:
                axs[kk]['area'].set_xticklabels(['' for mm in range(0,33,3)])
            else:
                axs[kk]['area'].set_xticklabels([str((dt.strptime('2016-06-01','%Y-%m-%d') + relativedelta(months=mm)))[0:7] for mm in range(0,33,3)])


            for col in ['cropland','forestshrub','grassy','human']:
                np.log10(dfs[kk][dfs[kk][col+'_orig']>0].area).hist(ax=axs[kk]['dist'], bins=15, alpha=0.75, edgecolor=self.colors_base[col],histtype='step', linewidth=2,density=True, orientation='horizontal', fill=False)

            

            axs[kk]['scatter'].set_ylabel('Area [$m^2$]')
            axs[kk]['area'].set_xlabel('')
            axs[kk]['scatter'].set_xlabel('')
            
            axs[kk]['area'].set_xlim(['2016-06-01','2018-10-31'])
            axs[kk]['scatter'].set_xlim([np.datetime64(dt.strptime('2016-06-01','%Y-%m-%d')).astype(np.int64)*1000, np.datetime64(dt.strptime('2018-10-31','%Y-%m-%d')).astype(np.int64)*1000])
            axs[kk]['scatter'].set_ylim([10**2, 10**7])
            axs[kk]['dist'].set_ylim([2, 7])
            axs[kk]['dist'].set_yticklabels([])
            axs[kk]['dist'].grid(False)
            axs[kk]['dist'].set_xticklabels([])

            axs[kk]['area'].yaxis.set_major_formatter(mtick.PercentFormatter(1))
            axs[kk]['scatter'].text(np.datetime64(dt.strptime('2018-09-01','%Y-%m-%d')).astype(np.int64)*1000,10**6.5,kk, fontsize=20)
            
            #axs[kk]['area'] = fig.add_subplot(gs[(ii_k%2)*4+3,ii_k//2])
            
            
            #print ([ts for ts in ax.get_xticks()])
            #axs[kk]['scatter'].set_xticklabels([dt.fromtimestamp(ts).strftime('%Y-%m') for ts in ax.get_xticks()])

        axs['IN']['scatter'].set_yticklabels([])
        axs['EU']['scatter'].set_yticklabels([])
        axs['IN']['scatter'].set_ylabel('')
        axs['EU']['scatter'].set_ylabel('')
        axs['IN']['area'].set_yticklabels([])
        axs['EU']['area'].set_yticklabels([])
            
        custom_lines = []
        for kk,vv in self.colors_dict.items():
            custom_lines.append(Line2D([0],[0],color=vv,marker='.',linestyle=None, lw=0))


        lgd = fig.legend(custom_lines, self.colors_dict.keys(), ncol=6, bbox_to_anchor=(0.5, 0.15), loc='center',fancybox=True)

        plt.savefig('./land_cover_regional.png', bbox_extra_artists=(lgd,))
        plt.show()

if __name__=="__main__":
    generator=Figure()
    generator.make_global()
