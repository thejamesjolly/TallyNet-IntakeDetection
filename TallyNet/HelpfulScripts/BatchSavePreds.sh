# ####
# Script to train many TallyNet Models with various architectures
####

DetMethod=1
EvalMethod=1
WinTol=99

# sbatch batch_script_TallyNet_SavePreds.sh FolderName DatabaseFlag DetectMethod EvalMethod WinTol WinSize


sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS15_s05_0" 3 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS15_s05_1" 3 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS15_s05_2" 3 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS15_s05_3" 3 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS15_s05_4" 3 $DetMethod $EvalMethod $WinTol 15

# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS10_s05_0" 3 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS10_s05_1" 3 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS10_s05_2" 3 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS10_s05_3" 3 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS10_s05_4" 3 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS20_s05_0" 3 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS20_s05_1" 3 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS20_s05_2" 3 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS20_s05_3" 3 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS20_s05_4" 3 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS25_s05_0" 3 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS25_s05_1" 3 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS25_s05_2" 3 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS25_s05_3" 3 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS25_s05_4" 3 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS5_s05_0" 3 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS5_s05_1" 3 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS5_s05_2" 3 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS5_s05_3" 3 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DndClemson_A1333_WS5_s05_4" 3 $DetMethod $EvalMethod $WinTol 05



sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS15_s03_0" 1 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS15_s03_1" 1 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS15_s03_2" 1 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS15_s03_3" 1 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS15_s03_4" 1 $DetMethod $EvalMethod $WinTol 15

# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS10_s03_0" 1 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS10_s03_1" 1 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS10_s03_2" 1 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS10_s03_3" 1 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS10_s03_4" 1 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS20_s03_0" 1 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS20_s03_1" 1 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS20_s03_2" 1 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS20_s03_3" 1 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS20_s03_4" 1 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS25_s03_0" 1 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS25_s03_1" 1 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS25_s03_2" 1 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS25_s03_3" 1 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS25_s03_4" 1 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS5_s03_0" 1 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS5_s03_1" 1 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS5_s03_2" 1 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS5_s03_3" 1 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DndOHO_A1333_WS5_s03_4" 1 $DetMethod $EvalMethod $WinTol 05


sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS15_s03_0" 2 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS15_s03_1" 2 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS15_s03_2" 2 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS15_s03_3" 2 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS15_s03_4" 2 $DetMethod $EvalMethod $WinTol 15

# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS10_s03_0" 2 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS10_s03_1" 2 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS10_s03_2" 2 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS10_s03_3" 2 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS10_s03_4" 2 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS20_s03_0" 2 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS20_s03_1" 2 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS20_s03_2" 2 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS20_s03_3" 2 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS20_s03_4" 2 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS25_s03_0" 2 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS25_s03_1" 2 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS25_s03_2" 2 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS25_s03_3" 2 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS25_s03_4" 2 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS5_s03_0" 2 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS5_s03_1" 2 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS5_s03_2" 2 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS5_s03_3" 2 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DndTHO_A1333_WS5_s03_4" 2 $DetMethod $EvalMethod $WinTol 05

sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS15_s05_0" 5 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS15_s05_1" 5 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS15_s05_2" 5 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS15_s05_3" 5 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS15_s05_4" 5 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS15_s05_5" 5 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS15_s05_6" 5 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS15_s05_7" 5 $DetMethod $EvalMethod $WinTol 15

# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS10_s05_0" 5 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS10_s05_1" 5 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS10_s05_2" 5 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS10_s05_3" 5 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS10_s05_4" 5 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS10_s05_5" 5 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS20_s05_0" 5 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS20_s05_1" 5 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS20_s05_2" 5 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS20_s05_3" 5 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS20_s05_4" 5 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS20_s05_5" 5 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS25_s05_0" 5 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS25_s05_1" 5 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS25_s05_2" 5 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS25_s05_3" 5 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS25_s05_4" 5 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS25_s05_5" 5 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS5_s05_0" 5 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS5_s05_1" 5 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS5_s05_2" 5 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS5_s05_3" 5 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS5_s05_4" 5 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DomClemson_A1333_WS5_s05_5" 5 $DetMethod $EvalMethod $WinTol 05

sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS15_s03_0" 4 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS15_s03_1" 4 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS15_s03_2" 4 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS15_s03_3" 4 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS15_s03_4" 4 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS15_s03_5" 4 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS15_s03_6" 4 $DetMethod $EvalMethod $WinTol 15
sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS15_s03_7" 4 $DetMethod $EvalMethod $WinTol 15

# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS10_s03_0" 4 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS10_s03_1" 4 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS10_s03_2" 4 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS10_s03_3" 4 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS10_s03_4" 4 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS10_s03_5" 4 $DetMethod $EvalMethod $WinTol 10
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS20_s03_0" 4 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS20_s03_1" 4 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS20_s03_2" 4 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS20_s03_3" 4 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS20_s03_4" 4 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS20_s03_5" 4 $DetMethod $EvalMethod $WinTol 20
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS25_s03_0" 4 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS25_s03_1" 4 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS25_s03_2" 4 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS25_s03_3" 4 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS25_s03_4" 4 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS25_s03_5" 4 $DetMethod $EvalMethod $WinTol 25
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS5_s03_0" 4 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS5_s03_1" 4 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS5_s03_2" 4 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS5_s03_3" 4 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS5_s03_4" 4 $DetMethod $EvalMethod $WinTol 05
# sbatch batch_script_TallyNet_SavePreds.sh "DomOHO_A1333_WS5_s03_5" 4 $DetMethod $EvalMethod $WinTol 05



















# sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_0" 2 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_1" 2 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_2" 2 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_3" 2 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s02_0" 2 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s03_2" 2 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s03_3" 2 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s03_4" 2 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s03_5" 2 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s03_6" 2 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s05_0" 2 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s05_1" 2 1 1 99 15




# sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_0" 1 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_1" 1 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_2" 1 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_3" 1 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s02_0" 1 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s03_3" 1 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s03_4" 1 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s03_5" 1 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s03_6" 1 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s05_0" 1 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s05_1" 1 1 1 99 15
# sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s05_2" 1 1 1 99 15



for Filters in {3..3}
do
	for Lstm in {4..4}
	do
		for Dense in {3..3}
		do
			archSelection=$Filters$Lstm$Dense
			### EXAMPLE ### 
			# ./batch_script_TallyNet_SavePreds.sh "ModelFolderName" "Database_Flag" "Det_Method" "Eval_Method_Flag" "WinTol"
			# echo "\nScheduling Eval of A1"$archSelection"_0 Five Fold"
			# sbatch ./batch_script_TallyNet_SavePreds.sh "Clem_A1"$archSelection"_0" 3 $DetMethod $EvalMethod $WinTol
			# sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1"$archSelection"_0" 1 $DetMethod $EvalMethod $WinTol
			# sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1"$archSelection"_0" 2 $DetMethod $EvalMethod $WinTol
			
# 			echo "\nScheduling Eval of W25_A1"$archSelection"_0 Five Fold"
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "Clem_W25_A1"$archSelection"_s05_0" 3 $DetMethod $EvalMethod $WinTol 25
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_W25_A1"$archSelection"_s10_0" 1 $DetMethod $EvalMethod $WinTol 25
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "THO_W25_A1"$archSelection"_s10_0" 2 $DetMethod $EvalMethod $WinTol 25
			
# 			echo "\nScheduling Eval of W40A1"$archSelection"_0 Five Fold"
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "Clem_W40_A1"$archSelection"_s05_0" 3 $DetMethod $EvalMethod $WinTol 40
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_W40_A1"$archSelection"_s10_0" 1 $DetMethod $EvalMethod $WinTol 40
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "THO_W40_A1"$archSelection"_s10_0" 2 $DetMethod $EvalMethod $WinTol 40

            
# 			echo "\nScheduling Eval of A1"$archSelection"_0 Five Fold"
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "Clem_A1"$archSelection"_s05_2" 3 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1"$archSelection"_s05_2" 1 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1"$archSelection"_s05_2" 2 $DetMethod $EvalMethod $WinTol
			
			
# 			echo "\nScheduling Eval of A1"$archSelection"_1 Five Fold"
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "Clem_A1"$archSelection"_s05_1" 3 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1"$archSelection"_s05_1" 1 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1"$archSelection"_s05_1" 2 $DetMethod $EvalMethod $WinTol
			
			
# 			echo "\nScheduling Eval of A1"$archSelection"_2 Five Fold"
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "Clem_A1"$archSelection"_2" 3 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1"$archSelection"_2" 1 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1"$archSelection"_2" 2 $DetMethod $EvalMethod $WinTol
			
# 			echo "\nScheduling Eval of A1"$archSelection"_3 Five Fold"
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "Clem_A1"$archSelection"_3" 3 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1"$archSelection"_3" 1 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1"$archSelection"_3" 2 $DetMethod $EvalMethod $WinTol
			
			# echo "\nScheduling Eval of A1"$archSelection"_4 Five Fold"
			# sbatch ./batch_script_TallyNet_SavePreds.sh "Clem_A1"$archSelection"_4" 3 $DetMethod $EvalMethod $WinTol
			# sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1"$archSelection"_4" 1 $DetMethod $EvalMethod $WinTol
			# sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1"$archSelection"_4" 2 $DetMethod $EvalMethod $WinTol
			
# 			echo "\nScheduling Eval of A1"$archSelection"_5 Five Fold"
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "Clem_A1"$archSelection"_5" 3 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1"$archSelection"_5" 1 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1"$archSelection"_5" 2 $DetMethod $EvalMethod $WinTol
		done
	done
done


<<YET_TO_RUN



sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_0" 2 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_1" 2 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_2" 2 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_3" 2 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s02_0" 2 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s03_2" 2 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s03_3" 2 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s03_4" 2 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s03_5" 2 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s03_6" 2 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s05_0" 2 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "THO_A1333_s05_1" 2 1 1 99




sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_0" 1 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_1" 1 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_2" 1 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_3" 1 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s02_0" 1 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s03_3" 1 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s03_4" 1 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s03_5" 1 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s03_6" 1 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s05_0" 1 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s05_1" 1 1 1 99
sbatch ./batch_script_TallyNet_SavePreds.sh "OHO_A1333_s05_2" 1 1 1 99


YET_TO_RUN

