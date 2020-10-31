### Calcate energy and greenhouse gas emissions associate with model training and deployment

DEPLOY_GCP_CPU = 1000000 #hours
DEPLOY_GCP_GPU = 20000 # hours

TRAIN_AWS_GPU = 24*5 # hours
TRAIN_GCP_GPU = 24*5 # hours

ENERGY_CPU = 65  #W
ENERGY_GPU = 300 #W

EM_RATE_GCP = 0 # kg/kwh
EM_RATE_AWS = 0.3715 + 0.0175 # kg/kWh


EN_GCP = DEPLOY_GCP_CPU * ENERGY_CPU + (TRAIN_GCP_GPU+DEPLOY_GCP_GPU) * ENERGY_GPU #Wh
EN_AWS = TRAIN_AWS_GPU * ENERGY_GPU #Wh


TOT_ENERGY = (EN_GCP+EN_AWS) / 1000 / 1000 # MWh

TOT_EM = EN_AWS/1000*EM_RATE_AWS # kg


ELECCAR = 0.2 #kWh/km
PETCAR = 0.125 #kg/km

print (f'Total energy: {TOT_ENERGY} MWh')
print (f'... is eq to {TOT_ENERGY*1000/ELECCAR} km by electric car.')
print (f'TOTAL emissions: {TOT_EM} kg')
print (f'... is equivalent to {TOT_EM/PETCAR} km in a petrol car')
