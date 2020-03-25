import yaml

print (yaml.__version__)

probability_band = {
      'name':'probability',
      'type':'mask',
      'srcband':1,
      'dtype':'Byte',
      'nbits':8,
      'data_range':[0,255],
      'colormap_name':'magma'
}

dd = {}
dd['S2-V1-Primary'] ={
            'cloud_id':None,
            'name':'solar_pv:s2_v1:primary_pass',
            'title':'S2 UNet Primary FeatureCollection',
            'description':'Solar PV vectorisation from the inital pass of the S2 UNet model.',
            'type':'vector',
            'other':None,
            'timestamp':None
        }
dd['S2-V2-Secondary'] ={
            'cloud_id':None,
            'name':'solar_pv:s2_v2:secondary_pass',
            'title':'S2 UNet Secondary FeatureCollection',
            'description':'Solar PV vectorisation based on the filter by RNN-1.',
            'type':'vector',
            'other':None,
            'timestamp':None,
        }
dd['S2-V3-Deepstack'] ={
            'cloud_id':None,
            'name':'solar_pv:s2_v3:deepstack',
            'title':'S2 UNet Deepstack FeatureCollection',
            'description':'Solar PV vectorisation based on the deepstack inference.',
            'type':'vector',
            'other':None,
            'timestamp':None
        }
dd['S2-V4-Final'] ={
            'cloud_id':None,
            'name':'solar_pv:s2_v4:final',
            'title':'S2 UNet Final FeatureCollection',
            'description':'Solar PV vectorisation after RNN-2 filtering.',
            'type':'vector',
            'other':None,
            'timestamp':None
        }
dd['S2-R1-Primary'] ={
            'cloud_id':None,
            'name':'solar_pv:s2_r1:primary_pass',
            'title':'S2 UNet Primary Raster',
            'description':'Solar PV inference from the inital pass of the S2 UNet model.',
            'type':'raster',
            'other':{'bands':[probability_band]},
            'timestamp':None
        }
dd['S2-R2-Secondary'] ={
            'cloud_id':None,
            'name':'solar_pv:s2_r2:secondary_pass',
            'title':'S2 UNet Secondary Raster',
            'description':'Solar PV inference after the filter by RNN-1.',
            'type':'raster',
            'other':{'bands':[probability_band]},
            'timestamp':None,
        }
dd['SPOT-V1-Vectorised'] ={
            'cloud_id':None,
            'name':'solar_pv:spot_v1:vectorisation',
            'title':'SPOT 6/7 Vectorised FeatureCollection',
            'description':'Solar PV vectorised from the primary SPOT UNet pass.',
            'type':'vector',
            'other':None,
            'timestamp':None
        }

dd['SPOT-R1-Primary'] ={
            'cloud_id':None,
            'name':'solar_pv:spot_r1:primary_pass',
            'title':'SPOT 6/7 Primary Raster',
            'description':'Solar PV inference from the initial pass of the SPOT UNet model.',
            'type':'raster',
            'other':{'bands':[probability_band]},
            'timestamp':None
        }

dd['SPOT-V2-Filtered'] ={
            'cloud_id':None,
            'name':'solar_pv:spot_v2:filtered',
            'title':'SPOT 6/7 Filtered FeatureCollection',
            'description':'Solar PV featurecollection filtered by ResNet50.',
            'type':'vector',
            'other':None,
            'timestamp':None
        }

dd['Combined-V1'] ={
            'cloud_id':None,
            'name':'solar_pv:combined_v1:vectorisation',
            'title':'Solar PV Combined FeatureCollection',
            'description':'Solar PV featurecollection combining S2 and SPOT pipelines.',
            'type':'vector',
            'other':None,
            'timestamp':None
        }



yaml.dump(dd,open('cloud_products.yaml','w'))

# set the products actually used
dd['S2-V1-Primary']['cloud_id'] = 'ba611607613832ad7bb8fa9dc2bafb71f693bd6a:V1-solar_pv-S2-20190308-primary_pass'
dd['S2-V2-Secondary']['cloud_id'] = 'ba611607613832ad7bb8fa9dc2bafb71f693bd6a:V2-solar_pv-S2-20190327-secondary_filter'
dd['S2-V3-Deepstack']['cloud_id'] = 'ba611607613832ad7bb8fa9dc2bafb71f693bd6a:V3-solar_pv-S2-20190328-deepstack'
dd['S2-V4-Final']['cloud_id'] = 'ba611607613832ad7bb8fa9dc2bafb71f693bd6a:v4-solar_pv-s2-20190322_final'
dd['S2-R1-Primary']['cloud_id'] = 'ba611607613832ad7bb8fa9dc2bafb71f693bd6a:solar_pv_S2_v3_20190306_primary'
dd['S2-R2-Secondary']['cloud_id'] = 'ba611607613832ad7bb8fa9dc2bafb71f693bd6a:solar_pv:s2:v3_20190327:secondary'
dd['SPOT-V1-Vectorised']['cloud_id'] = 'ba611607613832ad7bb8fa9dc2bafb71f693bd6a:v1b-solar-pv-spot'
dd['SPOT-R1-Primary']['cloud_id'] = '8514dad6c277e007cedb6fb8e829a23c8975fca4:solar_pv:airbus:spot:v5_0111'
dd['SPOT-V2-Filtered']['cloud_id'] = 'ba611607613832ad7bb8fa9dc2bafb71f693bd6a:v1-solar_pv-spot-20200317-postfilter'
dd['Combined-V1']['cloud_id'] = 'ba611607613832ad7bb8fa9dc2bafb71f693bd6a:v1-solar-pv-20190521-combined-final'

yaml.dump(dd, open('cloud_products_exec.yaml','w'))