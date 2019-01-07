import sys
import csv
import pandas as pd
import glob
import numpy as np
from math import exp, ceil

test_path = '/dataset_testset_directory/test_set/'
test_input = sorted(glob.glob(test_path + "log_input*.csv"))

pred_path = '/testset_predictions_base_directory/'
pred_input = sorted(glob.glob(pred_path + "*.txt"))

output_fname = '/direcotry_final_submission_file/'
missing_output_file = '/directory_for_missing_files/missing.txt'

def generate_submission(predictions, pred_file):
    output = []
    with open(output_fname+pred_file, 'w') as fout:
        for i,f_test in enumerate(test_input):
            df_test = pd.read_csv(f_test)
            last_session = None
            for track_row in df_test.itertuples():
                curr_session = track_row.session_id
                if last_session != curr_session:
                    if curr_session not in predictions:
                        print ("missing %d" % i)
                        output.append("%s,%d" % (curr_session, i))
                        print('1'*int(ceil(track_row.session_length/2)), file=fout, flush=True)
                    else:
                        print(predictions[curr_session], file=fout, flush=True)
                    last_session = curr_session
    with open(missing_output_file, 'w') as fout:
        for i in output:
            print(i, file=fout, flush=True)

if __name__ == "__main__":
    predictions = {}
    for file_ in pred_input:
        csvreader = csv.reader(open(file_))
        for row in csvreader:
            if row[1] != "":
                if row[0] not in predictions:
                    last_act = 1.0
                    if row[-1] == 'False':
                        last_act = 0.0
                    pred = [[]]
                    counter = 0 
                    for p in row[1:-1]:
                        if counter == 10:
                            counter = 0
                            pred.append([])
                        pred[-1].append(p)
                        counter += 1
                    if '_2.txt' in file_:
                        pred = pred[::-1]
                    final_pred = ""
                    len_pred = len(pred)
                    for i, p in enumerate(pred):
                        curr_div = 0.0
                        curr_pred = 0.0
                        first_pred = 0
                        last_pred = 0
                        for j, p2 in enumerate(p):
                            if j<len_pred:
                                curr_div += (len_pred*2 - 1) - abs(i*2-j*2)
                                curr_pred += float(p2)* ((len_pred*2 - 1) - abs(i*2-j*2))
                        curr_pred += last_act*len_pred*2
                        curr_div += len_pred*2
                        final_pred += str(int(round(curr_pred / curr_div)))
                    predictions[row[0]] = final_pred
                elif (len(predictions[row[0]])*10) != len(row[1:-1]):
                    print(row[0], ",", file_)
    generate_submission(predictions, "submission_final.txt")


