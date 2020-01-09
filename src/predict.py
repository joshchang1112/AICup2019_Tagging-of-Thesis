import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
from metrics import Recall
import torch
import pandas as pd

def main(args):
    # load config
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)


    # make model
    if config['arch'] == 'BertNet':
        from example_predictor import ExamplePredictor
        PredictorClass = ExamplePredictor

    predictor = PredictorClass(metrics=[],
                               **config['model_parameters'])
    model_path = os.path.join(
        args.model_dir,
        'model.pkl.{}'.format(args.epoch))
    logging.info('loading model from {}'.format(model_path))
    predictor.load(model_path)

    # predict test
    logging.info('loading test data...')
    #with open('../data/valid.pkl', 'rb') as f:
    with open(config['test'], 'rb') as f:
        test = pickle.load(f)
        test.shuffle = False
    logging.info('predicting...')
    predicts = predictor.predict_dataset(test, test.collate_fn)

    output_path = os.path.join(args.model_dir,
                               'predict-{}.csv'.format(args.epoch))
    write_predict_csv(predicts, test, output_path)


def write_predict_csv(predicts, data, output_path, n=10):
    print(predicts.size())
    sample = pd.read_csv('../data/task1_sample_submission.csv')
    threshold = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    predicts = predicts.cpu()
    count = 262948 - 131166
    answer = torch.zeros(count, 6)
    for i in range(count):
        tmp = 0
        #if i % 100 == 0:
        #    print(i) 
        for j in range(6):
            
            if predicts[i][j] >= threshold[j]:
                answer[i][j] = 1
                tmp = 1
            else:
                #sample.iloc[i, j+1] = 0
                answer[i][j] = 0
           # sample.iloc[i, j+1] = float(predicts[i][j])
        
        if tmp == 0:
            _, max_id = torch.max(predicts[i].unsqueeze(0), dim=1)
            #print(max_id)
            #sample.iloc[i, int(max_id)+1] = 1
            #sample.iloc[i, 4] = 1
            answer[i][max_id] = 1
    
    #sample = sample.iloc[:20000, ] 
    #sample['THEORETICAL'] = predict_label[:, 0]
    #sample['ENGINEERING'] = predict_label[:, 1]
    #sample['EMPIRICAL'] = predict_label[:, 2]
    #sample['OTHERS'] = predict_label[:, 3]
    answer = answer.numpy()
    sample.iloc[131166:262948, 1] = answer[:, 0]
    sample.iloc[131166:262948, 2] = answer[:, 1]
    sample.iloc[131166:262948, 3] = answer[:, 2]
    sample.iloc[131166:262948, 4] = answer[:, 3]
    sample.iloc[131166:262948, 5] = answer[:, 4]
    sample.iloc[131166:262948, 6] = answer[:, 5]
    logging.info('Writing output to {}'.format(output_path))
    sample.to_csv('prediction.csv')

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('model_dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--device', default=None,
                        help='Device used to train. Can be cpu or cuda:0,'
                        ' cuda:1, etc.')
    parser.add_argument('--not_load', action='store_true',
                        help='Do not load any model.')
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
