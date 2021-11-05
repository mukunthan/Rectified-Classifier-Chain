import pandas as pd
from tqdm import tqdm
import os

genidlist=[]
LabGenID=[]

Labeldf = pd.read_csv('AMR_LAbel_Salmonella.csv', sep=",", dtype=str, index_col=0, low_memory=True)
selectedf = Labeldf[['genome_id']]
Antiboticslist=Labeldf.columns.values.tolist()
#Antiboticslist.remove('Unnamed: 0')
Antiboticslist.remove( 'genome_id')
Antiboticslist.remove('genome_name')

Labeldf = Labeldf.replace(to_replace=['Susceptible', 'Intermediate', 'Resistant','susceptible', 'intermediate', 'resistant'], value=[0, 0.5, 1,0, 0.5, 1])

for genid in selectedf.values:
    LabGenID.append(genid)

def PopulateAMRProteinMatrix(Dir,Filename,matrixdf):
    pd.options.display.float_format = '{:,.5f}'.format
    df = pd.read_csv(Dir + Filename, sep="\t", dtype=str, low_memory=True)
    genid=str((df['genome_id'].iloc[0]))
    #print (genid)
    #print(genidlist)
    if(genid in LabGenID) and (genid not in genidlist):

        matrixdf = matrixdf.append({'genome_id': genid, 'genome_name':(Labeldf[Labeldf['genome_id'] == genid])['genome_name'].values, 'taxon_id':(Labeldf[Labeldf['genome_id'] == genid])['taxon_id'].values},
                                   ignore_index=True)
        for Antiboitics in Antiboticslist:
            matrixdf.loc[matrixdf['genome_id'] == genid, Antiboitics] = (Labeldf[Labeldf['genome_id'] == genid])[Antiboitics].values

        selectedf = df[['genome_id', 'genome_name', 'plfam_id']]
        selectedf = selectedf.dropna(subset=['plfam_id'])
        genidlist.append(genid)
        for genid, gennam, plfam_id in selectedf.values:
            if plfam_id not in matrixdf.columns:
                finaldf2 = pd.DataFrame({plfam_id: [0]})
                matrixdf = matrixdf.join(finaldf2)

            matrixdf.loc[matrixdf['genome_id'] == genid, plfam_id] = 1
    elif (genid in LabGenID):
        print ('Duplicate')
    else:
        print ("Not in Lab based")
    return matrixdf

def ReadFromFeaturesFolders(path, Matdf: object):
    for filename in tqdm(os.listdir(path)):
        Matdf = PopulateAMRProteinMatrix(path, filename, Matdf)
    #Matdf.to_csv('Test_Multilabel_final.csv', sep=',')
    return Matdf


finaldf = pd.DataFrame({'genome_id':[''],'genome_name':[''],'taxon_id':[''],'ampicillin':[0], 'amoxicillin/clavulanic acid':[0], 'aztreonam':[0], 'cefepime':[0], 'cefotaxime':[0], 'cefoxitin':[0], 'ceftazidime':[0], 'ciprofloxacin':[0], 'gentamicin':[0], 'piperacillin/tazobactam':[0], 'sulfamethoxazole/trimethoprim':[0], 'tobramycin':[0], 'trimethoprim':[0]
})

dir='data_Salmonella/'
finaldf=ReadFromFeaturesFolders(dir,finaldf)
#####################
#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_id_Multilabel_Salmonella_data.csv',sep=',',float_format='%.5f')
################
