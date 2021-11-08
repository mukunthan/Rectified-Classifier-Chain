# Rectified-Classifier-Chain
Rectified Classifier Chain for Classifying Multi-Label Antibiotic Resistance Data with Missing Labels

Run Similar steps for Salmonella to get the resutls related to Salmonella 
RectifiedClassifierChain.py is the proposed model implementation
StackedClassifierChain.py is Stacked labels with subset correction for Classifierchain implementation

Prerequists
Numpy, Pandas, Shap, sklearn, XGBoost, skmultilearn, matplotlib, Seaborn, tqdm

Steps
1. Download PATRIC_genomes_AMR.txt
2. Run AMR_Label_Processing.py; It will create AMR_LAbel_Salmonella.csv and GenomeListEcoli.txt
3. Run DwnLdFromPATRIC_EColi.sh ; It will download all the features files required
4. Run DataPreprocessing_Multilabel_Ecoli.py; It will create Finalplfam_id_Multilabel_Ecoli_data.csv
5. Run Multi-Label_Missing-label_EColi.ipynb to get results with different models
6. Run Compare_With_SoA_Work.ipynb to get results with the SoA works
7. Copy the time values printed at the end of step 5 to Complexity_Time.py and Run to get the average time taken for each steps
8. Copy the Top featues saved in generated feature files in Step 5 and Run PrepareOutput.py to check the definition of features and known genes


