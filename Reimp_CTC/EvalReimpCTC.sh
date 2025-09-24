

DatasetFlag=$1

if [[ $DatasetFlag -eq 1 ]]; then

    python ReimpCTC_Eval.py 1 1 99 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODND_CTC_v1/
    python ReimpCTC_Eval.py 1 1 99 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODND_CTC_v2/
    python ReimpCTC_Eval.py 1 1 99 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODND_CTC_v3/
    python ReimpCTC_Eval.py 1 1 99 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODND_CTC_v4/

    echo ""
    python ReimpCTC_Eval.py 1 2  8 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODND_CTC_v1/
    python ReimpCTC_Eval.py 1 2  8 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODND_CTC_v2/
    python ReimpCTC_Eval.py 1 2  8 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODND_CTC_v3/
    python ReimpCTC_Eval.py 1 2  8 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODND_CTC_v4/




elif [[ $DatasetFlag -eq 2 ]]; then

    python ReimpCTC_Eval.py 2 1 99 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_THO_v1/
    python ReimpCTC_Eval.py 2 1 99 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_THO_v2/
    python ReimpCTC_Eval.py 2 1 99 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_THO_v3/
    python ReimpCTC_Eval.py 2 1 99 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_THO_v4/

    echo ""
    python ReimpCTC_Eval.py 2 2  8 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_THO_v1/
    python ReimpCTC_Eval.py 2 2  8 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_THO_v2/
    python ReimpCTC_Eval.py 2 2  8 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_THO_v3/
    python ReimpCTC_Eval.py 2 2  8 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_THO_v4/




elif [[ $DatasetFlag -eq 3 ]]; then

    python ReimpCTC_Eval.py 3 1 99 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_DNDClem_CTC_v1/
    python ReimpCTC_Eval.py 3 1 99 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_DNDClem_CTC_v2/
    python ReimpCTC_Eval.py 3 1 99 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_DNDClem_CTC_v3/
    python ReimpCTC_Eval.py 3 1 99 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_DNDClem_CTC_v4/

    echo ""
    python ReimpCTC_Eval.py 3 2  8 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_DNDClem_CTC_v1/
    python ReimpCTC_Eval.py 3 2  8 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_DNDClem_CTC_v2/
    python ReimpCTC_Eval.py 3 2  8 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_DNDClem_CTC_v3/
    python ReimpCTC_Eval.py 3 2  8 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_DNDClem_CTC_v4/


elif [[ $DatasetFlag -eq 4 ]]; then
    python ReimpCTC_Eval.py 4 1 99 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODom_CTC_v0/
    python ReimpCTC_Eval.py 4 1 99 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODom_CTC_v1/
    python ReimpCTC_Eval.py 4 1 99 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODom_CTC_v2/
    python ReimpCTC_Eval.py 4 1 99 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODom_CTC_v3/

    echo ""
    python ReimpCTC_Eval.py 4 2  8 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODom_CTC_v0/
    python ReimpCTC_Eval.py 4 2  8 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODom_CTC_v1/
    python ReimpCTC_Eval.py 4 2  8 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODom_CTC_v2/
    python ReimpCTC_Eval.py 4 2  8 /home/jpjolly/CTC_Rouast_Take2/oneHandOrebaAttempt/Preds_OHODom_CTC_v3/





elif [[ $DatasetFlag -eq 5 ]]; then

    python ReimpCTC_Eval.py 5 1 99 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_ClemDom_v1/
    python ReimpCTC_Eval.py 5 1 99 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_ClemDom_v2/
    python ReimpCTC_Eval.py 5 1 99 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_ClemDom_v3/
    python ReimpCTC_Eval.py 5 1 99 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_ClemDom_v4/

    echo ""
    python ReimpCTC_Eval.py 5 2  8 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_ClemDom_v1/
    python ReimpCTC_Eval.py 5 2  8 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_ClemDom_v2/
    python ReimpCTC_Eval.py 5 2  8 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_ClemDom_v3/
    python ReimpCTC_Eval.py 5 2  8 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_ClemDom_v4/


fi



# python ReimpCTC_Eval.py 1 1 99 ./FiveFold_CTCResults_Examples/Dnd_OHO_Intake/ 
# echo ""
# python ReimpCTC_Eval.py 1 2  8 ./FiveFold_CTCResults_Examples/Dnd_OHO_Intake/



# python ReimpCTC_Eval.py 2 1 99 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_THO_v4/
# echo ""
# python ReimpCTC_Eval.py 2 2  8 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_THO_v4/



# python ReimpCTC_Eval.py 3 1 99 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_DNDClem_CTC_v4/
# echo ""
# python ReimpCTC_Eval.py 3 2  8 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_DNDClem_CTC_v4/



# python ReimpCTC_Eval.py 4 1 99 ./FiveFold_CTCResults_Examples/Dom_OHO_Intake/
# echo ""
# python ReimpCTC_Eval.py 4 2  8 ./FiveFold_CTCResults_Examples/Dom_OHO_Intake/





# python ReimpCTC_Eval.py 5 1 99 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_ClemDom_v4/

# echo ""

# python ReimpCTC_Eval.py 5 2  8 /home/jpjolly/CTC_Rouast_Take2/ctc-intake-detection-master/Preds_ClemDom_v4/










