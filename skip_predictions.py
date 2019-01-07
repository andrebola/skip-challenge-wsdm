import sys
import pandas as pd
import glob
import math
import xgboost
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets import dump_svmlight_file


data_path = '/dataset_base_directory/'
training_path = data_path + 'training_set/'
input_logs = sorted(glob.glob(training_path + "*.csv"))

features_path = '/spotify_track_features_path/track_features/'

train_features_fname = '/directory_extracted_features/features_split/'

xgboost_model_location = '/base_directory_trained_models/xgboost_model'

test_path = '/dataset_testset_directory/test_set/'
test_prehistory = sorted(glob.glob(test_path + "log_prehistory*.csv")) 
test_input = sorted(glob.glob(test_path + "log_input*.csv")) 

output_fname = '/testset_predictions_base_directory/xgboost_'


features = ['us_popularity_estimate', 'acousticness', 'beat_strength', 'bounciness', 'danceability',
       'dyn_range_mean', 'energy', 'flatness', 'instrumentalness',
       'liveness', 'loudness', 'mechanism','tempo' , 'organism', 'speechiness', 'valence']


def extract_features_session(tracks):
    return tracks[features].mean().to_dict()


def extract_features_track(song):
    return song[features+["duration"]].to_dict()


def extract_features_session_track(song, positive_tracks, negative_tracks):
    mean_duration_pos = (positive_tracks['duration'] - song['duration']).mean()
    mean_duration_neg = (negative_tracks['duration'] - song['duration']).mean()
    
    year_pos = (positive_tracks['release_date_estimate'] - song['release_date_estimate']).mean()
    year_neg = (negative_tracks['release_date_estimate'] - song['release_date_estimate']).mean()
   
    pop_pos = (positive_tracks['us_popularity_estimate'] - song['us_popularity_estimate']).mean()
    pop_neg = (negative_tracks['us_popularity_estimate'] - song['us_popularity_estimate']).mean()
 
    latent_vectors = ['acoustic_vector_0', 'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3', 'acoustic_vector_4', 'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7']
    mean_dot_vector_pos = positive_tracks[latent_vectors].dot(song[latent_vectors]).mean()
    mean_dot_vector_neg = negative_tracks[latent_vectors].dot(song[latent_vectors]).mean()
 
    return {'pos_duration': mean_duration_pos, 
            'neg_duration': mean_duration_neg,
            'pos_year': year_pos,
            'neg_year': year_neg,
            'pop_neg': pop_neg,
            'pop_pos': pop_pos,
            'pos_dot': mean_dot_vector_pos,
            'neg_dot': mean_dot_vector_neg
           }



def get_ground_truth(f_input, tracks_features_df, i):
        output_f_name = train_features_fname+str(i)+".svm"
        try:
            num_lines_completed = sum(1 for line in open(output_f_name))
        except FileNotFoundError:
            num_lines_completed = 0

        dv = None
        tracks_features = []
        features_labels = []     
        #for i,f in enumerate(input_logs):
        df = pd.read_csv(f_input)
        # Below we keep only the relevant columns of the second half of the session for saving the ground truth
        df = df[['session_id','skip_2','session_position','session_length','track_id_clean', 'premium', 'not_skipped', 'hist_user_behavior_is_shuffle', 'hour_of_day', 'date']]
        current_index = 0
        completed_index = 0
        # Here we process each session, saving a list containing the 
        while current_index < len(df):
            partial_length = df['session_length'].iloc[current_index]-df['session_position'].iloc[current_index]+1
            last_session_tracks = df.loc[current_index+(partial_length/2):current_index+partial_length-1]
            if completed_index < num_lines_completed:
                current_index += partial_length 
                completed_index += len(last_session_tracks)
                continue

            first_session_tracks = df.loc[current_index:current_index+(partial_length/2)-1]
            last_session_item = first_session_tracks.iloc[-1]
                      
            skipped = first_session_tracks[first_session_tracks['skip_2'] == 1]['track_id_clean']
            completed = first_session_tracks[first_session_tracks['skip_2'] == 0]['track_id_clean']
            
            skipped = tracks_features_df.loc[skipped]
            completed = tracks_features_df.loc[completed]
            
            features_skipped = extract_features_session(skipped)
            features_completed = extract_features_session(completed)
            
            other_features = {}
            other_features['rskip'] = len(skipped) / float(len(first_session_tracks))
            other_features['rskip2'] = (first_session_tracks['not_skipped'] == 0).sum() / float(len(first_session_tracks))
            
            other_features["premium"] = last_session_item["premium"]
            other_features["shuffle"] = last_session_item["hist_user_behavior_is_shuffle"]
            other_features["hour"] = last_session_item["hour_of_day"]
            last_date = last_session_item["date"].split("-")
            other_features["day"] = int(last_date[2])
            other_features["month"] = int(last_date[1])           
            
            for j, session_track_row in last_session_tracks.iterrows():
                song_row = tracks_features_df.loc[session_track_row['track_id_clean']]
                track_features = extract_features_track(song_row)
                track_features["distance"] = session_track_row["session_position"] - last_session_item["session_position"]
                
                session_track_features = extract_features_session_track(song_row, completed, skipped)
                
                all_track_features = track_features
                all_track_features.update(session_track_features)
                
                all_track_features.update({k+'_neg': v for k, v in features_skipped.items()})
                all_track_features.update({k+'_pos': v for k, v in features_completed.items()})
                all_track_features.update(other_features)
                
                tracks_features.append({x:y for x,y in all_track_features.items() if not np.isnan(y)})
                features_labels.append(session_track_row['skip_2']) 
            current_index += partial_length 
            if current_index % 100 == 0:
                if dv is None:
                    dv = DictVectorizer()
                    X = dv.fit_transform(tracks_features)
                else:
                    X = dv.transform(tracks_features)
                y = np.array(features_labels)
                dump_svmlight_file(X, y, open(output_f_name, 'ab'))
                features_labels = []
                tracks_features = []
        #return tracks_features, features_labels
        if len(tracks_features):
            if dv is None:
                dv = DictVectorizer()
                X = dv.fit_transform(tracks_features)
            else:
                X = dv.transform(tracks_features)
            y = np.array(features_labels)
            dump_svmlight_file(X, y, open(output_f_name, 'ab'))
 
def train_xgboost(position_sk):
    xgtrain = xgboost.DMatrix(train_features_fname+str(position_sk)+".svm")
    
    params = {
        'objective':'binary:logistic',
        'eta':0.3, 
        'min_child_weight': 1,
        'max_depth':15,
        'subsample': 1,
        'colsample_bytree': 1.0,
        'tree_method': 'hist',
        'base_score': 0.1,
        'eval_metric': 'auc',
        'use_buffer': 0,
        'seed':1,
        'njobs':16
    }

    model = xgboost.train(
        dtrain=xgtrain,
        params=params,  
        num_boost_round=200 
    )
    model.save_model(xgboost_model_location+str(position_sk)+'.npz')


def generate_submission(f_test, f_history, i, get_ground_truth, models):
    dv = DictVectorizer().fit([{'us_popularity_estimate': 0., 'acousticness': 0., 'beat_strength': 0., 'bounciness': 0., 'danceability': 0., 'dyn_range_mean': 0., 'energy': 0., 'flatness': 0., 'instrumentalness': 0., 'liveness': 0., 'loudness': 0., 'mechanism': 0., 'tempo': 0., 'organism': 0., 'speechiness': 0., 'valence': 0., 'duration': 0., 'distance': 0, 'pos_duration': 0., 'neg_duration': 0., 'pos_year': 0., 'neg_year': 0., 'pop_neg': 0., 'pop_pos': 0., 'pos_dot': 0., 'neg_dot': 0., 'rskip': 0., 'rskip2': 0., 'premium': True, 'shuffle': False, 'hour': 0, 'day': 0, 'month': 0, 'us_popularity_estimate_neg': 0., 'acousticness_neg': 0., 'beat_strength_neg': 0., 'bounciness_neg': 0., 'danceability_neg': 0., 'dyn_range_mean_neg': 0., 'energy_neg': 0., 'flatness_neg': 0., 'instrumentalness_neg': 0., 'liveness_neg': 0., 'loudness_neg': 0., 'mechanism_neg': 0., 'tempo_neg': 0., 'organism_neg': 0., 'speechiness_neg': 0., 'valence_neg': 0., 'us_popularity_estimate_pos': 0., 'acousticness_pos': 0., 'beat_strength_pos': 0., 'bounciness_pos': 0., 'danceability_pos': 0., 'dyn_range_mean_pos': 0., 'energy_pos': 0., 'flatness_pos': 0., 'instrumentalness_pos': 0., 'liveness_pos': 0., 'loudness_pos': 0., 'mechanism_pos': 0., 'tempo_pos': 0., 'organism_pos': 0., 'speechiness_pos': 0., 'valence_pos': 0.}])

    try:
        completed_sessions = {line.split(",")[0]:1 for line in open(output_fname+str(i)+".txt")}
        completed_sessions.update({line.split(",")[0]:1 for line in open(output_fname+str(i)+"_2.txt")})
        print (len(completed_sessions))
    except FileNotFoundError:
        completed_sessions = {}

    #with open(output_fname+str(i)+"_2.txt", 'a') as fout:
    with open(output_fname+str(i)+".txt", 'a') as fout:
            df_test = pd.read_csv(f_test)
            #f_history = test_prehistory[i]
            df_history = pd.read_csv(f_history)
            print('file {} read'.format(i))
            # Below we keep only the relevant columns of the second half of the session for saving the ground truth
            df_history = df_history[['session_id','skip_2','session_position','session_length','track_id_clean', 'premium', 'not_skipped', 'hist_user_behavior_is_shuffle', 'hour_of_day', 'date']]
            df_history.set_index('session_id', inplace=True)

            last_session = None
            features_skipped = None
            features_completed = None
            other_features = None
            session_tracks = None
            last_session_item = None

            output = []
            # Here we process each session, saving a list containing the predictions
            #for _, track_row in df_test[::-1].iterrows():
            for _, track_row in df_test.iterrows():
                curr_session = track_row['session_id']
                if curr_session in completed_sessions :
                    if len(output) > 0:
                        line = last_session+","+','.join(map(str,output))
                        print(line, file=fout, flush=True)
                        output = []
                    last_session = curr_session
                    continue
                if last_session != curr_session:
                    if len(output) > 0:
                        output.append(last_session_item['skip_2'])
                        line = last_session+","+','.join(map(str,output))
                        print(line, file=fout, flush=True)
                        output = []
                    last_session = curr_session

                    session_tracks = df_history.loc[last_session]

                    skipped = session_tracks[session_tracks['skip_2'] == 1]['track_id_clean']
                    completed = session_tracks[session_tracks['skip_2'] == 0]['track_id_clean']

                    skipped = tracks_features_df.loc[skipped]
                    completed = tracks_features_df.loc[completed]

                    features_skipped = extract_features_session(skipped)
                    features_completed = extract_features_session(completed)

                    last_session_item = session_tracks.iloc[-1]

                    other_features = {}
                    other_features['rskip'] = len(skipped) / float(len(session_tracks))
                    other_features['rskip2'] = (session_tracks['not_skipped'] == 0).sum() / float(len(session_tracks))

                    other_features["premium"] = last_session_item["premium"]
                    other_features["shuffle"] = last_session_item["hist_user_behavior_is_shuffle"]
                    other_features["hour"] = last_session_item["hour_of_day"]
                    last_date = last_session_item["date"].split("-")
                    other_features["day"] = int(last_date[2])
                    other_features["month"] = int(last_date[1])   

                song_row = tracks_features_df.loc[track_row['track_id_clean']]
                track_features = extract_features_track(song_row)
                distance = track_row["session_position"] - last_session_item["session_position"]
                track_features["distance"] = distance

                session_track_features = extract_features_session_track(song_row, completed, skipped)
                all_track_features = track_features
                all_track_features.update(session_track_features)
                all_track_features.update(other_features)

                all_track_features.update({k+'_neg': v for k, v in features_skipped.items()})
                all_track_features.update({k+'_pos': v for k, v in features_completed.items()})

                pred_X_feat = dv.transform(all_track_features)
                dfeat = xgboost.DMatrix(pred_X_feat)

                for model in models:
                    score = model.predict(dfeat)[0]
                    output.append(score)
            output.append(last_session_item['skip_2'])
            line = last_session+","+','.join(map(str,output))
            print(line, file=fout, flush=True)

if __name__ == "__main__":
    num_workers = 4
    offset = 0
    step = ''
    if len (sys.argv) > 1 :
        offset = int(sys.argv[2])
        step = sys.argv[1]
    
    all_files = glob.glob(features_path + "*.csv")

    tracks_features_df = pd.DataFrame()
    list_ = []
    for file_ in all_files:
        df = pd.read_csv(file_,index_col=None, header=0)
        list_.append(df)
    tracks_features_df = pd.concat(list_)
    tracks_features_df.set_index('track_id', inplace=True)

    tracks_features_df['release_date_estimate'] = pd.to_numeric(tracks_features_df['release_date_estimate'].str[0:4])
    
    
    if 'features':
        pool = ThreadPool(num_workers)

        for i,f_input in enumerate(input_logs[offset:offset+num_workers]):
            pool.apply_async(get_ground_truth, args=(f_input, tracks_features_df, i+offset))

        pool.close()
        pool.join()

    elif 'train':
        train_xgboost(offset)

    elif 'predict':
        models = []
        for j in range(10):
            model = xgboost.Booster()  # init model
            model.load_model(xgboost_model_location+str(j+1)+".npz")  # load data
            models.append(model)

        pool = ThreadPool(num_workers)

        for i,f_test in enumerate(test_input[offset:offset+num_workers]):
            f_history = test_prehistory[i+offset]
            pool.apply_async(generate_submission, args=(f_test, f_history, i+offset, tracks_features_df, models))

        pool.close()
        pool.join()
