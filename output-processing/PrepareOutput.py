import pandas as pd

import os
'''
plflist1=[ 'PLF_561_00017967','PLF_561_00193238','PLF_561_00001373', 'PLF_561_00009406','PLF_561_00004782',
'PLF_561_00077715','PLF_561_00005154','PLF_561_00009167','PLF_561_00003224','PLF_561_00006018','PLF_561_00002933',
'PLF_561_00014654','PLF_561_00001270','PLF_561_00011189','PLF_561_00015082','PLF_561_00007970','PLF_561_00005960',
'PLF_561_00003871','PLF_561_00003559','PLF_561_00003099','PLF_561_00003317','PLF_561_00005992','PLF_561_00003405',
'PLF_561_00003864','PLF_561_00196761','PLF_561_00010837','PLF_561_00012028''PLF_561_00084507']
'''
plflist1=['PLF_561_00175318','PLF_561_00017967','PLF_561_00005659','PLF_561_00193238','PLF_561_00003864',
          'PLF_561_00069189','PLF_561_00003159','PLF_561_00013716','PLF_561_00009406','PLF_561_00071721',
          'PLF_561_00004782','PLF_561_00007970','PLF_561_00022754','PLF_561_00019473','PLF_561_00011189',
          'PLF_561_00049890','PLF_561_00077715','PLF_561_00027964','PLF_561_00009167','PLF_561_00008661',
          'PLF_561_00098623','PLF_561_00001373','PLF_561_00014654']

def CheckGenes(dir,plflist,finaldf):
    if (len(plflist) > 0):
        for filename in os.listdir(dir):
            df = pd.read_csv(dir + filename, sep="\t", index_col=5, low_memory=False)
            selectedf = df[['plfam_id', 'pgfam_id', 'feature_type', 'gene', 'product']]
            selectedf = selectedf.dropna(subset=['plfam_id'])
            if (len(plflist) <= 0):
                break
            for itm in plflist:
                pgfam = selectedf[selectedf['plfam_id'] == itm]
                if (len(pgfam.values) != 0):
                    print('found')
                    finaldf = finaldf.append({'plfam_id': itm, 'pgfam_id': pgfam['pgfam_id'].iloc[0],
                                              'feature_type': pgfam['feature_type'].iloc[0],
                                              'gene': pgfam['gene'].iloc[0], 'product': pgfam['product'].iloc[0]},
                                             ignore_index=True)
                    plflist.remove(itm)
    return plflist,finaldf


plflist=list(set(plflist1))
finaldf = pd.DataFrame({'plfam_id':[''],'pgfam_id':[''],'feature_type':[''],'figfam_id':[''],'gene':[''], 'product':['']})

dir='../dataset/Escherichia/amoxicillin/Resistant/features/'
plflist,finaldf=CheckGenes(dir,plflist,finaldf)

dir='../dataset/Escherichia/amoxicillin/Susceptible/features/'
plflist,finaldf=CheckGenes(dir,plflist,finaldf)

dir='../dataset/Escherichia/ampicillin/Resistant/features/'
plflist,finaldf=CheckGenes(dir,plflist,finaldf)

dir='../dataset/Escherichia/ampicillin/Susceptible/features/'
plflist,finaldf=CheckGenes(dir,plflist,finaldf)

dir='../dataset/Escherichia/aztreonam/Resistant/features/'
plflist,finaldf=CheckGenes(dir,plflist,finaldf)

dir='../dataset/Escherichia/aztreonam/Susceptible/features/'
plflist,finaldf=CheckGenes(dir,plflist,finaldf)

dir='../dataset/Escherichia/cefepime/Resistant/features/'
plflist,finaldf=CheckGenes(dir,plflist,finaldf)

dir='../dataset/Escherichia/cefepime/Susceptible/features/'
plflist,finaldf=CheckGenes(dir,plflist,finaldf)

dir='../dataset/Escherichia/cefotaxime/Resistant/features/'
plflist,finaldf=CheckGenes(dir,plflist,finaldf)

dir='../dataset/Escherichia/cefotaxime/Susceptible/features/'
plflist,finaldf=CheckGenes(dir,plflist,finaldf)

print(len(plflist))
for element in plflist:
    print (element)
finaldf.to_csv('plfam_id_description_dataset_xgb_final.csv',sep=',')




