import pandas as pd

import os

plflist1=['PLF_561_00049890','PLF_561_00004782','PLF_561_00007970','PLF_561_00005659','PLF_561_00006091',
          'PLF_561_00013716','PLF_561_00019473','PLF_561_00009406','PLF_561_00015082','PLF_561_00003350',
          'PLF_561_00009585','PLF_561_00005191','PLF_561_00008661','PLF_561_00019538','PLF_561_00008369',
          'PLF_561_00000117','PLF_561_00071721','PLF_561_00006969','PLF_561_00005448','PLF_561_00028701',
          'PLF_561_00017967','PLF_561_00028114','PLF_561_00005992','PLF_561_00011189','PLF_561_00175318',
          'PLF_561_00028965','PLF_561_00003914','PLF_561_00051514','PLF_561_00009167']

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
dir='data_Ecoli/'
plflist,finaldf=CheckGenes(dir,plflist,finaldf)
        
print(len(plflist))
for element in plflist:
    print (element)
finaldf.to_csv('../output/plfam_id_description_dataset_xgb_EColi.csv',sep=',')




