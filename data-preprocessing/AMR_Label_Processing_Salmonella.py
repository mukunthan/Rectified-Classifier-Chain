import pandas as pd

genomeList=[]
def ReadAMRFileandPopulate(matrixdf,selecteddf,genomeList):

    for genid, genomename, taxon, type, AMR  in selecteddf.values:
        if (genid in genomeList):
            matrixdf.loc[matrixdf['genome_id'] == genid, type] = AMR
        else:
            genomeList.append(genid)
            matrixdf = matrixdf.append(
                {'genome_id': genid, 'genome_name': genomename, 'taxon_id':taxon, type: AMR}, ignore_index=True)
    return matrixdf,genomeList



dir='../data/'
AMRTextFileName='PATRIC_genomes_AMR.txt'
df = pd.read_csv(dir + AMRTextFileName, sep="\t", dtype=str, low_memory=True)
print(df.shape)
df=df[df['genome_name'].str.contains("Salmonella")]
df = df.dropna(subset=['resistant_phenotype'])

selectedf=df[['genome_id','genome_name','taxon_id','antibiotic','resistant_phenotype']]
selectedf=selectedf.reset_index(drop=True)

Matrixdf = pd.DataFrame({'genome_id':[''],'genome_name':[''],'taxon_id':['']})
Matrixdf, genomeList=ReadAMRFileandPopulate(Matrixdf,selectedf,genomeList)

Matrixdf = Matrixdf.iloc[1: , :]

print(Matrixdf.shape)
######Dropping AMR which has less than 200 Lab based experiments

DropColList=[]
for col in Matrixdf.columns :
    count=Matrixdf[col].isnull().sum()
    if(count >= Matrixdf.shape[0]-200):
        DropColList.append(col)


Matrixdf=Matrixdf.drop(DropColList, axis=1)
Matrixdf=Matrixdf.reset_index(drop=True)

ListofRowtoDrop=[]
## After dropping the Fewer AMR columns, check the row and remove if there are Genomes with no Labels
for i in range(len(Matrixdf.index)) :
    rowNoncount=Matrixdf.iloc[i].isnull().sum()
    if(rowNoncount >= Matrixdf.shape[1]-3):  ### £ here to avoid, Genome ID, Genome Name, Taxon ID columns
        print(" Total NaN in row", i + 1, ":",
          rowNoncount)
        selectedgenid=(Matrixdf.iloc[i])['genome_id']
        genomeList.remove(selectedgenid)
        ListofRowtoDrop.append(i)

Matrixdf = Matrixdf.drop(ListofRowtoDrop,axis=0)
Matrixdf=Matrixdf.reset_index(drop=True)
Matrixdf.to_csv(dir+'AMR_LAbel_Salmonella.csv',sep=',')
print(Matrixdf.shape)
print(Matrixdf.columns.values)

f= open('../data/Genome_List_Salmonella.txt', 'w')
for item in genomeList:
    f.write(item+ "\n")
f.close()

