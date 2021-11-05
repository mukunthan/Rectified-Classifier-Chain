list=$(cat Genome_List_Salmonella.txt)
#echo "$list";
for i in $list
do 
 #echo "$i";
 wget -qN "ftp://ftp.patricbrc.org/genomes/$i/$i.PATRIC.features.tab" ;
done
