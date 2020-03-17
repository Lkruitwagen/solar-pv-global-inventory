import yaml

print (yaml.__version__)

dd = {}
dd['S2Infer1'] ={
            'image': "us.gcr.io/dl-ci-cd/images/tasks/public/py3.7-gpu:v2020.01.21-7-g309186be",
            'name':"S2Infer1",
            'requirements':[],
            'maximum_concurrency':500,
            'memory':'7.4Gi',
            'retry_count':0,
            'task_timeout':21600,
            'group_id':None,
            'timestamp':None,
            'tiledict':{'tilesize':2000,'resolution':10,'pad':0}
        }

dd['S2Infer2'] = {
            'image':"us.gcr.io/dl-ci-cd/images/tasks/public/py3.7-gpu:v2020.01.21-7-g309186be",
            'name':"S2Infer2",
            'requirements': [],
            'maximum_concurrency': 500,
            'memory':'3.5Gi',
            'retry_count':0,
            'task_timeout':21600,
            'group_id':None,
            'timestamp':None,
            'tiledict':None
}

dd['S2RNN1'] = {
            'image' : "us.gcr.io/dl-ci-cd/images/tasks/public/py3.7-gpu:v2020.01.21-7-g309186be",
            'name':"S2RRN1",
            'requirements' : [],
            'maximum_concurrency':15,
            'memory':'3.5Gi',
            'retry_count':0,
            'task_timeout':21600,
            'group_id':None,
            'timestamp':None,
            'tiledict':{'tilesize':10000,'resolution':10,'pad':0}
}

dd['SPOTVectoriser'] = {
            'image' :"us.gcr.io/dl-ci-cd/images/tasks/public/py3.7-gpu:v2020.01.21-7-g309186be",
            'name':"SPOTVectoriser",
            'requirements' : [],
            'maximum_concurrency':200,
            'memory':'3.7Gi',
            'retry_count':0,
            'task_timeout':28800,
            'group_id':None,
            'timestamp':None,
            'tiledict':{'tilesize':4096,'resolution':1.5,'pad':200}
}

yaml.dump(dd,open('cloud_functions.yaml','w'))