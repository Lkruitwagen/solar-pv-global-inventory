import geojson
import geopandas as gpd
import descarteslabs as dl
import pandas as pd
from shapely import geometry
from shapely.ops import cascaded_union
import numpy as np
import json, os, sys
import yaml
import pickle
import matplotlib.pyplot as plt
gpd.options.use_pygeos=False

from matplotlib.collections import LineCollection
import matplotlib.ticker as mtick

plt.style.use('ggplot')

def hex2rgb(h):
    h = h.lstrip('#')
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]


root = os.getcwd()

res_dict = pickle.load(open(os.path.join(root,'data','res_dict_10k.pickle'),'rb'))# open('../../data/res_dict.pkl','rb'))
iou_dict = pickle.load(open(os.path.join(root,'data','iou_dict_10k.pickle'),'rb'))# open('../../data/iou_dict.pickle','rb'))

for key, vv in res_dict.items():
    for ar,vv2 in vv.items():
        vv2['iou'] = iou_dict[key][ar]['iou']
        #vv2['iou_neg'] = iou_dict[key][ar]['iou_neg']
        #vv2['iou_pos'] = iou_dict[key][ar]['iou_pos']

del res_dict['pre-handlabel']

title_dict = {'P':'Precision','R':'Recall','iou':'Intersection-over-Union'}

gg_colors = [tuple(ih/255 for ih in hex2rgb(ii['color'])) for ii in list(plt.rcParams['axes.prop_cycle'])[0:3]]

area_bins = [1e4, 1e10]

fig, axs = plt.subplots(len(area_bins)-1,3,figsize=(18,4),sharey=True, sharex=True)
axs = axs.reshape((1,-1))


for ii_a in range(len(area_bins)-1):
    
    for ii_ax, M in enumerate(['P','R','iou']):
        full_bars= [res_dict[kk][ii_a][M] for kk in res_dict.keys()] 

        bars = [full_bars[0]] + \
            [(full_bars[ii] - full_bars[ii-1]) for ii in range(1,4)] + \
            [full_bars[4], full_bars[5]-full_bars[4]] +\
            [full_bars[6]]
        bottoms = [0]+\
                    [full_bars[ii-1] for ii in range(1,4)] +\
                    [0,full_bars[4],0]
        
        lines_y = [el for el in full_bars for _ in (0,1)]
        lines_x = [0] + [el for el in range(1,6) for _ in (0,1)] + [6]
        segs = [[[lines_x[ii], lines_y[ii]],[lines_x[ii+1],lines_y[ii+1]]] for ii in range(0,12,2)]
        
        segs[3][1][0]=3.5
        segs[5][1][0]=5.5
        segs.append([segs[3][1],[segs[3][1][0],full_bars[6]]])
        segs.append([[segs[3][1][0],full_bars[6]],[6,full_bars[6]]])
        segs.append([segs[5][1],[segs[5][1][0],full_bars[6]]])
        
        
        line_segments = LineCollection(segs, colors=[gg_colors[0]]*4 + [gg_colors[1]]*2 + [gg_colors[2]]*3, alpha=0.5)
        axs[ii_a,ii_ax].add_collection(line_segments)

        colors = [gg_colors[0]]*4 + [gg_colors[1]]*2 + [gg_colors[2]]
        print (len(res_dict.keys()))
        print (len(bars))
        print (len(bottoms))


        axs[ii_a,ii_ax].bar(range(len(res_dict.keys())),bars, bottom=bottoms, edgecolor=colors, linewidth=2,color=colors)
        
        for ii in range(7):
            H=0.05
            if ((M=='R' and ii_a==0) or (M=='iou' and ii_a)):
                H=.1
                
            axs[ii_a,ii_ax].text(ii,H,f'{full_bars[ii]:.0%}', horizontalalignment='center')

        
        axs[ii_a,ii_ax].set_xticklabels(['','S1-V1','S1-V2','S1-V3','S1-V4','SPOT-V1','SPOT-V2','Final'])
        
        if ii_a==0:
            axs[ii_a,ii_ax].set_title(title_dict[M],fontsize=24)
            axs[ii_a,ii_ax].set_ylim([0,1])
            


    axs[ii_a,0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
        
    #axs[ii_a,0].set_ylabel(f'Installation area > 10,000m$^2$')
    
fig.savefig(os.path.join(root,'makefigs','figures','fig-A6_P-R-iou_single.png'))