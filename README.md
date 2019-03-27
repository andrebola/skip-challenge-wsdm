


# Spotify Sequential Skip Prediction Challenge 2019 - WSDM Cup workshop

This work was develop in the [Music Techology Group](https://www.upf.edu/web/mtg), under the supervision of Xavier Serra.

**Team**: aferraro

**Members**: 

 - Andres Ferraro
 - Dmitry Bogdanov

**contact:** andres.ferraro at upf.edu

Instructions
-----------

In order to reproduce the solution submitted for the challenge the following steps must be executed:

##### 1. compute features for all the training set:

With the script `skip_predictions.py` we compute the features for the training set, for this the first parameter is 'features' and the second parameter is the training file to compute ([0-659]), we have to call this script for all the files. By the default the script processes 4 files so if we call:

```skip_predictions.py features 0```

It will process files from 0 to 3. The idea is to call this process from different nodes (for different files) so we can distribute the load. 

Before execting the script you must change the location of the required files (training-set:`training_path` and spotify track features: `features_path`). Also, you need to specify the location of the output files, for this step you only need to specify the location for the extracted features for training models (`train_features_fname`).

##### 2. split features in 10 files depending on the position of the track:

Once we extracted all the features for the training set we have to divide the training examples in 10 different files, one for each position. For this we use the following awk command:

```
awk -F "\"* \"*" '{for (i=1; i <= 65; i++) { split($i,a,":"); if (a[1] == '13') print >> (a[2]".svm")}}' /directory_of_extrated_features/*.svm
```

##### 3. train one model for each training file:

Once we have 10 files with the training examples we have to run the `skip_predictions.py` script to train the models. For this we use 'train' as the first parameter and the second parameter is the file number that we will use [1-10]:

```skip_predictions.py train 1```

Before executing the script is important to specify the loaction where the trained models will be placed (`xgboost_model_location`). Also, make sure that the location of the 10 files with the training examples is correct (`train_features_fname`).

##### 4. compute prediction for test set:

Once we have all the models we compute the predictions for all the test set. For this we also use the script `skip_predictions.py`, this time the first parameter is 'predict' and the second parameter is the file that we want to process [0-65]. This works in the same way as the step 1, so we can call the process from diferent machines using different offsets, by default each call processes 4 files:

```skip_predictions.py predict 0```

Before executing the script make sure to specify the location of the test-set (`test_path`) and the folder where the predictions will be saved (`output_fname`)

##### 5. create submission file:

Finally we create the submission file using the script `create_submission.py`. Inside the script we have to spocify the location of the predictions (output of step 4), and the location of the test_set.


Citations
-----

For citations you can reference this paper::

Andrés Ferraro, Dmitry Bogdanov, and Xavier Serra. **"Skip prediction using boosting trees based on acoustic features of tracks in sessions"**. WSDM Cup 2019 Workshop on the 12th ACM International Conference on Web Search and Data Mining, February 15, 2019, Malbourne, Australia.
