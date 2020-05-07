import json, logging, pickle, os, sys, time

from shapely import geometry
from pulp import *
import pandas as pd
import geopandas as gpd
import numpy as np
from geopy.distance import geodesic

tic = time.time()


from shapely.strtree import STRtree

import networkx as nx

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def milp_geodesic_network_satisficing(pts_A, pts_B, alpha,mipgap=0.0001,v=False):
    pts_A_dict = {pt.name:pt for pt in pts_A}
    pts_B_dict = {pt.name:pt for pt in pts_B}
    
    A_names = [pt.name for pt in pts_A]
    B_names = [pt.name for pt in pts_B]
    
    Z = {pt.name:{} for pt in pts_A}
    MW_A = {pt.name:pt.MW for pt in pts_A}
    MW_B = {pt.name:pt.MW for pt in pts_B}
    
    if v:
        print ('generating Z..')
    for pt_A in pts_A:
        for pt_B in pts_B:
            
            Z[pt_A.name][pt_B.name]=(geodesic([pt_A.y,pt_A.x], [pt_B.y,pt_B.x]).kilometers)**2
            
    sum_Z = sum([Z[A_name][B_name] for A_name in A_names for B_name in B_names])
    
    ### declare model
    model = LpProblem("Network Satisficing Problem",LpMinimize)
    
    ### Declare variables
    # B -> Bipartite Network
    B = LpVariable.dicts("Bipartite",(A_names,B_names),0,1,LpInteger)
    
    # abs_diffs -> absolute value forcing variable
    abs_diffs = LpVariable.dicts("abs_diffs",B_names,cat='Continuous')
    
    ### Declare constraints
    # Contstraint - abs diffs edges
    for B_name in B_names:
        model += abs_diffs[B_name] >= (MW_B[B_name] - lpSum([MW_A[A_name]*B[A_name][B_name] for A_name in A_names]))/MW_B[B_name],"abs forcing pos {}".format(B_name) 
        model += abs_diffs[B_name] >= -1 * (MW_B[B_name] - lpSum([MW_A[A_name]*B[A_name][B_name] for A_name in A_names]))/MW_B[B_name], "abs forcing neg {}".format(B_name)
        
    # Constraint - bipartite edges
    for A_name in A_names:
        model += lpSum([B[A_name][B_name] for B_name in B_names]) <= 1,"Bipartite Edges {}".format(A_name)
        
    ### Affine equations
    # Impedence error
    E_z = sum([Z[A_name][B_name]*B[A_name][B_name] for A_name in A_names for B_name in B_names])/sum_Z
    
    # mw error
    E_mw = sum([abs_diffs[B_name] for B_name in B_names])/len(B_names)
    
    ### Objective function
    model += E_z*alpha + (1-alpha)*E_mw, "Loss"
    if v:
        print ('solving model...')
    
    model.solve(pulp.GUROBI_CMD(options=[('MIPGap',str(mipgap)),('OutputFlag', str(0))]))
    if v:
        print(pulp.LpStatus[model.status])
    
    return model, B, E_z, E_mw, Z

class MatchRegion:

    def __init__(self, match,region, dist_buffer=10000, alpha=0.15, mipgap=0.0001):
        self.buffer=dist_buffer
        self.alpha=alpha
        self.mipgap=mipgap

        self.outpath = os.path.join(os.getcwd(),'data','_'.join(['match',match,'-'.join(region),str(dist_buffer),str(alpha)])+'.gpkg')

        if match=='wri':
            self.target_gdf= self.prep_wri()
        elif match=='eia':
            self.target_gdf = self.prep_eia()

        
        self.source_gdf = gpd.read_file(os.path.join(os.getcwd(),'data','ABCD_landcover.geojson')).reset_index()
        self.source_gdf['capacity_mw'] = self.source_gdf['area']*44.2/1000/1000
        self.source_gdf = self.source_gdf.rename(columns={'index':'unique_id'})
        self.source_gdf.geometry = self.source_gdf.geometry.representative_point()
        self.source_gdf = self.source_gdf[['unique_id','iso-3166-1','capacity_mw','geometry']]
        

        if region: # must be list(iso-3166-1), for now.
            self.source_gdf = self.source_gdf[self.source_gdf['iso-3166-1'].isin(region)]
            self.target_gdf = self.target_gdf[self.target_gdf['iso-3166-1'].isin(region)]

        print (self.source_gdf)
        print (self.target_gdf)
        

    def prep_wri(self):
        """
        output: gdf with lat/lon point geom, iso-3166-1, a unique_id, and MW capacity
        """
        wri = pd.read_csv(os.path.join(os.getcwd(),'data','WRI_gppd.csv'))
        iso2 = pd.read_csv(os.path.join(os.getcwd(),'data','iso2.csv'))

        # attach country iso-3166-1 to wri
        wri = wri.merge(iso2[['iso2','iso3']], how='left',left_on='country',right_on='iso3')

        #filter solar PV
        wri = wri[wri['fuel1']=='Solar']

        # rename iso2
        wri = wri.rename(columns={'iso2':'iso-3166-1','gppd_idnr':'unique_id'})

        # combine coordinates
        wri['coordinates'] = wri[['longitude','latitude']].values.tolist()

        # convert to shapely obj
        wri['geometry'] = wri['coordinates'].map(geometry.Point)

        wri = gpd.GeoDataFrame(wri[['unique_id','iso-3166-1','capacity_mw']], geometry=wri['geometry'], crs={'init':'epsg:4326'})

        return wri




    def prep_eia(self):
        """
        output: gdf with lat/lon point geom, iso-3166-1, a unique_id, and MW capacity
        """
        # load file
        eia = gpd.read_file(os.path.join(os.getcwd(),'data','eia_powerstations','PowerPlants_US_202001.shp'))

        # add iso-3166-1
        eia['iso-3166-1'] = 'US'

        # filter solar
        eia = eia[eia['PrimSource']=='solar']

        # rename cols
        eia = eia.rename(columns={'Plant_Code':'unique_id','Install_MW':'capacity_mw'})

        # keep cols
        eia = eia[['geometry','unique_id','capacity_mw','iso-3166-1']]

        return eia

    def get_components(self, dist_buffer=10000):
        """
        use source and target gdfs to create a network, get the connected components of the network
        """
        source_gdf = self.source_gdf.to_crs({'init':'epsg:3395'})
        source_gdf['3395_x'] = source_gdf.geometry.x
        source_gdf['3395_y'] = source_gdf.geometry.y

        target_gdf = self.target_gdf.to_crs({'init':'epsg:3395'})
        target_gdf['3395_x'] = target_gdf.geometry.x
        target_gdf['3395_y'] = target_gdf.geometry.y

        G = nx.Graph()

        source_df = pd.DataFrame(source_gdf)
        target_df = pd.DataFrame(target_gdf)



        def _match_st(row):
            bool_ind = (((target_df['3395_x'] - row['3395_x'])**2 + (target_df['3395_y'] - row['3395_y'])**2)**(1/2))<self.buffer
            return target_df[bool_ind]['unique_id'].values.tolist()

        source_df['sjoin_ids'] = source_df.apply(lambda row: _match_st(row), axis=1)
        print (source_df[source_df['sjoin_ids'].str.len()<1])

        for ii_r,row in enumerate(source_df.iterrows()):

            if ii_r %100==0: 
                print (ii_r)


            G.add_edges_from([(row[1]['unique_id'],ii) for ii in row[1]['sjoin_ids']])

        print ('n nodes', len(G.nodes))
        print ('e dges', len(G.edges))

        self.G = G
        


    def run_main(self):

        print ('n connect components', len([_ for _ in nx.connected_components(self.G)]))

        self.source_gdf['match_id'] = ''

        for ii_c, cc in enumerate(nx.connected_components(self.G)):
            
            B, E_z, E_mw = self.run_component(cc)

            matches = 0

            for kk, vv in B.items():
                for kk2, vv2 in vv.items():
                    if vv2.value()>0:
                        matches+=1
                        self.source_gdf.loc[kk,'match_id']=kk2

            print ('component',ii_c, len(cc), 'found matches:',matches)



        self.source_gdf.to_file(self.outfile, driver="GPKG")



    def run_component(self,component):

        source_slice = self.source_gdf[self.source_gdf['unique_id'].isin(list(component))]
        target_slice = self.target_gdf[self.target_gdf['unique_id'].isin(list(component))]
        
        pts_source = []
        pts_target = []

        for row in source_slice.iterrows():
            pt=row[1]['geometry']
            pt.name = row[1]['unique_id']
            pt.MW = row[1]['capacity_mw']
            pts_source.append(pt)

        for row in target_slice.iterrows():
            pt=row[1]['geometry']
            pt.name = row[1]['unique_id']
            pt.MW = row[1]['capacity_mw']
            pts_target.append(pt)

        model, B, E_z, E_mw, Z = milp_geodesic_network_satisficing(pts_source, pts_target, self.alpha, self.mipgap, v=False)

        return B, E_z, E_mw


if __name__=="__main__":
    matcher=MatchRegion('wri',['GB'], dist_buffer=10000, alpha=0.15,mipgap=0.002)
    matcher.get_components()
    matcher.run_main()
