import pandas as pd
from tqdm import tqdm
import os

genidlist=[]

def PopulateAMRProteinMatrix(Dir,Filename,matrixdf,Antibiotics,Phenotypes):
    df = pd.read_csv(Dir + Filename, sep="\t", index_col=5, dtype=str, low_memory=True)
    genid=str((df['genome_id'].iloc[0]))
    #print (genid)
    #print(genidlist)
    if (genid in genidlist):
        matrixdf.loc[matrixdf['genome_id'] == genid, Antibiotics] = Phenotypes
        print ('Already there')
    else:
        selectedf = df[['genome_id', 'genome_name', 'plfam_id']]
        selectedf = selectedf.dropna(subset=['plfam_id'])
        i = 0
        genidlist.append(genid)
        for genid, gennam, plfam_id in selectedf.values:
            if plfam_id not in matrixdf.columns:
                finaldf2 = pd.DataFrame({plfam_id: [0]})
                matrixdf = matrixdf.join(finaldf2)

            if (i == 0):
                matrixdf = matrixdf.append({'genome_id': genid, 'genome_name': gennam, Antibiotics: Phenotypes, plfam_id: 1},
                                         ignore_index=True)
                i = i + 1
            else:
                matrixdf.loc[matrixdf['genome_id'] == genid, plfam_id] = 1
                i = i + 1
    return matrixdf


finaldf = pd.DataFrame({'genome_id':[''],'genome_name':['']})


dir='../dataset/Escherichia/amoxicillin/Resistant/features/'
finaldf2 = pd.DataFrame({'amoxicillin': [0]})
finaldf = finaldf.join(finaldf2)
print ('amoxicillin -Resistant')

##tqdm#
for filename in tqdm(os.listdir(dir)):
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'amoxicillin', 1)

#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')
################################


dir='../dataset/Escherichia/amoxicillin/Susceptible/features/'
print ('amoxicillin -Susceptible')
for filename in tqdm(os.listdir(dir)):
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'amoxicillin', 0)

#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

dir='../dataset/Escherichia/ampicillin/Resistant/features/'
print ('ampicillin -Resistant')
finaldf2 = pd.DataFrame({'ampicillin': [0]})
finaldf = finaldf.join(finaldf2)
##tqdm#
for filename in tqdm(os.listdir(dir)):
    #print(filename)
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'ampicillin', 1)

#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

dir='../dataset/Escherichia/ampicillin/Susceptible/features/'
print ('ampicillin -Susceptible')
for filename in tqdm(os.listdir(dir)):
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'ampicillin', 0)


#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

####################
dir='../dataset/Escherichia/aztreonam/Resistant/features/'
print ('aztreonam -Resistant')
finaldf2 = pd.DataFrame({'aztreonam': [0]})
finaldf = finaldf.join(finaldf2)
##tqdm#
for filename in tqdm(os.listdir(dir)):
    #print(filename)
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'aztreonam', 1)

#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

dir='../dataset/Escherichia/aztreonam/Susceptible/features/'
print ('aztreonam -Susceptible')
for filename in tqdm(os.listdir(dir)):
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'aztreonam', 0)


#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

#####################
dir='../dataset/Escherichia/cefepime/Resistant/features/'
print ('cefepime -Resistant')
finaldf2 = pd.DataFrame({'cefepime': [0]})
finaldf = finaldf.join(finaldf2)
##tqdm#
for filename in tqdm(os.listdir(dir)):
    #print(filename)
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'cefepime', 1)

#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

dir='../dataset/Escherichia/cefepime/Susceptible/features/'
print ('cefepime -Susceptible')
for filename in tqdm(os.listdir(dir)):
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'cefepime', 0)


#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

####################################


dir='../dataset/Escherichia/cefotaxime/Resistant/features/'
print ('cefotaxime -Resistant')
finaldf2 = pd.DataFrame({'cefotaxime': [0]})
finaldf = finaldf.join(finaldf2)
##tqdm#
for filename in tqdm(os.listdir(dir)):
    #print(filename)
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'cefotaxime', 1)

#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

dir='../dataset/Escherichia/cefotaxime/Susceptible/features/'
print ('cefotaxime -Susceptible')
for filename in tqdm(os.listdir(dir)):
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'cefotaxime', 0)


#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')
################################

dir='../dataset/Escherichia/cefoxitin/Resistant/features/'
print ('cefoxitin -Resistant')
finaldf2 = pd.DataFrame({'cefoxitin': [0]})
finaldf = finaldf.join(finaldf2)
##tqdm#
for filename in tqdm(os.listdir(dir)):
    #print(filename)
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'cefoxitin', 1)

#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

dir='../dataset/Escherichia/cefoxitin/Susceptible/features/'
print ('cefoxitin -Susceptible')
for filename in tqdm(os.listdir(dir)):
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'cefoxitin', 0)


#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')
###########################

dir='../dataset/Escherichia/ceftazidime/Resistant/features/'
print ('ceftazidime -Resistant')
finaldf2 = pd.DataFrame({'ceftazidime': [0]})
finaldf = finaldf.join(finaldf2)
##tqdm#
for filename in tqdm(os.listdir(dir)):
    #print(filename)
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'ceftazidime', 1)

#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

dir='../dataset/Escherichia/ceftazidime/Susceptible/features/'
print ('ceftazidime -Susceptible')
for filename in tqdm(os.listdir(dir)):
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'ceftazidime', 0)


#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')
################


dir='../dataset/Escherichia/cefuroxime/Resistant/features/'
print ('cefuroxime -Resistant')
finaldf2 = pd.DataFrame({'cefuroxime': [0]})
finaldf = finaldf.join(finaldf2)
##tqdm#
for filename in tqdm(os.listdir(dir)):
    #print(filename)
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'cefuroxime', 1)

#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

dir='../dataset/Escherichia/cefuroxime/Susceptible/features/'
print ('cefuroxime -Susceptible')
for filename in tqdm(os.listdir(dir)):
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'cefuroxime', 0)


#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')
################

dir='../dataset/Escherichia/ciprofloxacin/Resistant/features/'
print ('ciprofloxacin -Resistant')
finaldf2 = pd.DataFrame({'ciprofloxacin': [0]})
finaldf = finaldf.join(finaldf2)
##tqdm#
for filename in tqdm(os.listdir(dir)):
    #print(filename)
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'ciprofloxacin', 1)

#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

dir='../dataset/Escherichia/ciprofloxacin/Susceptible/features/'
print ('ciprofloxacin -Susceptible')
for filename in tqdm(os.listdir(dir)):
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'ciprofloxacin', 0)


#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')
################

dir='../dataset/Escherichia/gentamicin/Resistant/features/'
print ('gentamicin -Resistant')
finaldf2 = pd.DataFrame({'gentamicin': [0]})
finaldf = finaldf.join(finaldf2)
##tqdm#
for filename in tqdm(os.listdir(dir)):
    #print(filename)
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'gentamicin', 1)

#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

dir='../dataset/Escherichia/gentamicin/Susceptible/features/'
print ('gentamicin -Susceptible')
for filename in tqdm(os.listdir(dir)):
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'gentamicin', 0)


#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')


##########
'''
#####################
##Use this if stop in the middle by commenting above parts
finaldf = pd.read_csv('Finalplfam_iddataset_Multilabel.csv',dtype=str, low_memory=True, index_col=0)
selectedf=finaldf['genome_id']
for values in selectedf.values:
    genidlist.append(str(values))
    #print (str(values))
#####################
'''
dir='../dataset/Escherichia/piperacillin/tazobactam/Resistant/features/'
print ('piperacillin -Resistant')
finaldf2 = pd.DataFrame({'piperacillin': [0]})
finaldf = finaldf.join(finaldf2)
##tqdm#
for filename in tqdm(os.listdir(dir)):
    #print(filename)
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'piperacillin', 1)

#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

dir='../dataset/Escherichia/piperacillin/tazobactam/Susceptible/features/'
print ('piperacillin -Susceptible')
for filename in tqdm(os.listdir(dir)):
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'piperacillin', 0)


#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')
################

dir='../dataset/Escherichia/tobramycin/Resistant/features/'
print ('tobramycin -Resistant')
finaldf2 = pd.DataFrame({'tobramycin': [0]})
finaldf = finaldf.join(finaldf2)
##tqdm#
for filename in tqdm(os.listdir(dir)):
    #print(filename)
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'tobramycin', 1)

#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

dir='../dataset/Escherichia/tobramycin/Susceptible/features/'
print ('tobramycin -Susceptible')
for filename in tqdm(os.listdir(dir)):
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'tobramycin', 0)


#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')
################

dir='../dataset/Escherichia/trimethoprim/Resistant/features/'
print ('trimethoprim -Resistant')
finaldf2 = pd.DataFrame({'trimethoprim': [0]})
finaldf = finaldf.join(finaldf2)
##tqdm#
for filename in tqdm(os.listdir(dir)):
    #print(filename)
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'trimethoprim', 1)

#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

dir='../dataset/Escherichia/trimethoprim/Susceptible/features/'
print ('trimethoprim -Susceptible')
for filename in tqdm(os.listdir(dir)):
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'trimethoprim', 0)

finaldf.to_csv('Finalplfam_iddataset_Multilabel_final.csv',sep=',')
##################
dir='../dataset/Escherichia/trimethoprim/sulfamethoxazole/Resistant/features/'
print ('sulfamethoxazole -Resistant')
finaldf2 = pd.DataFrame({'sulfamethoxazole': [0]})
finaldf = finaldf.join(finaldf2)
##tqdm#
for filename in tqdm(os.listdir(dir)):
    #print(filename)
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'sulfamethoxazole', 1)

#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel.csv',sep=',')

dir='../dataset/Escherichia/trimethoprim/sulfamethoxazole/Susceptible/features/'
print ('sulfamethoxazole -Susceptible')
for filename in tqdm(os.listdir(dir)):
    finaldf=PopulateAMRProteinMatrix(dir, filename, finaldf, 'sulfamethoxazole', 0)
#####################
#finaldf = finaldf.fillna(0)
finaldf.to_csv('Finalplfam_iddataset_Multilabel_final.csv',sep=',')
################
