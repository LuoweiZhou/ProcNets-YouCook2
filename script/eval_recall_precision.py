# --------------------------------------------------------
# Dense-Captioning Events in Videos Eval
# Copyright (c) 2017 Ranjay Krishna
# Licensed under The MIT License [see LICENSE for details]
# Written by Ranjay Krishna
# Modified by Luowei Zhou on Feb. 15, 2018
# --------------------------------------------------------

import argparse
import json
import sys

from sets import Set
import numpy as np

def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

class ANETcaptions(object):
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filenames=None, prediction_filename=None,
                 tious=None, max_proposals=1000,
                 prediction_fields=PREDICTION_FIELDS, split='validation', verbose=False):
        # Check that the gt and submission files exist and load them
        if len(tious) == 0:
            raise IOError('Please input a valid tIoU.')
        if not ground_truth_filenames:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')

        self.split = split
        self.verbose = verbose
        self.tious = tious
        self.max_proposals = max_proposals
        self.pred_fields = prediction_fields
        self.ground_truths = self.import_ground_truths(ground_truth_filenames)
        self.prediction = self.import_prediction(prediction_filename)
        # self.tokenizer = PTBTokenizer()

        # Set up scorers, if not verbose, we only use the one we're
        # testing on: METEOR
        """
        if self.verbose:
            self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(),"METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr")
            ]
        else:
            self.scorers = [(Meteor(), "METEOR")]
        """

    def import_prediction(self, prediction_filename):
        if self.verbose:
            print "| Loading submission..."
        submission = json.load(open(prediction_filename))
        if not all([field in submission.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid ground truth file.')
        # Ensure that every video is limited to the correct maximum number of proposals.
        results = {}
        for vid_id in submission['results']:
            results[vid_id] = submission['results'][vid_id][:self.max_proposals]
        return results

    def import_ground_truths(self, filenames):
        gts = []
        self.n_ref_vids = Set()
        for filename in filenames:
            gt = json.load(open(filename))['database']
            gt_split = {key:ann for key, ann in gt.iteritems() if ann['subset'] == self.split}
            self.n_ref_vids.update(gt_split.keys())
            gts.append(gt_split)
        if self.verbose:
            print "| Loading GT. #files: %d, #videos: %d" % (len(filenames), len(self.n_ref_vids))
        return gts

    def iou(self, interval_1, interval_2):
        start_i, end_i = interval_1[0], interval_1[1]
        start, end = interval_2[0], interval_2[1]
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
        iou = float(intersection) / (union + 1e-8)
        return iou

    def check_gt_exists(self, vid_id):
        for gt in self.ground_truths:
            if vid_id in gt:
              return True
        return False

    def get_gt_vid_ids(self):
        vid_ids = set([])
        for gt in self.ground_truths:
            vid_ids |= set(gt.keys())
        return list(vid_ids)

    def evaluate(self):
        self.scores = {}
        self.scores['Recall'] = []
        self.scores['Precision'] = []
        for tiou in self.tious:
            precision, recall = self.evaluate_detection(tiou)
            self.scores['Recall'].append(recall)
            self.scores['Precision'].append(precision)

    def evaluate_detection(self, tiou):
        gt_vid_ids = self.get_gt_vid_ids()
        # Recall is the percentage of ground truth that is covered by the predictions
        # Precision is the percentage of predictions that are valid
        recall = [0] * len(gt_vid_ids)
        precision = [0] * len(gt_vid_ids)
        for vid_i, vid_id in enumerate(gt_vid_ids):
            best_recall = 0
            best_precision = 0
            for gt in self.ground_truths:
                if vid_id not in gt:
                    continue
                refs = gt[vid_id]
                ref_set_covered = set([])
                pred_set_covered = set([])
                num_gt = 0
                num_pred = 0
                if vid_id in self.prediction:
                    for pred_i, pred in enumerate(self.prediction[vid_id]):
                        pred_timestamp = pred['segment']
                        for ref_i, ref_timestamp in enumerate(refs['annotations']):
                            if self.iou(pred_timestamp, ref_timestamp['segment']) > tiou:
                                ref_set_covered.add(ref_i)
                                pred_set_covered.add(pred_i)

                    new_precision = float(len(pred_set_covered)) / (pred_i + 1)
                    best_precision = max(best_precision, new_precision)
                new_recall = float(len(ref_set_covered)) / len(refs['annotations'])
                best_recall = max(best_recall, new_recall)
            recall[vid_i] = best_recall
            precision[vid_i] = best_precision
        return sum(precision) / len(precision), sum(recall) / len(recall)
        # return sum(recall) / len(recall), sum(precision) / len(precision)

def main(args):
    # Call coco eval
    evaluator = ANETcaptions(ground_truth_filenames=args.references,
                             prediction_filename=args.submission,
                             tious=args.tious,
                             max_proposals=args.max_proposals_per_video,
                             split=args.split,
                             verbose=args.verbose)
    evaluator.evaluate()

    # Output the results
    if args.verbose:
        for i, tiou in enumerate(args.tious):
            print '-' * 80
            print "tIoU: " , tiou
            print '-' * 80
            for metric in evaluator.scores:
                score = evaluator.scores[metric][i]
                print '| %s: %2.4f'%(metric, 100*score)

    # Print the averages
    print '-' * 80
    print "Average across all tIoUs"
    print '-' * 80
    for metric in evaluator.scores:
        score = evaluator.scores[metric]
        print '| %s: %2.4f'%(metric, 100 * sum(score) / float(len(score)))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate the results stored in a submissions file.')
    parser.add_argument('-s', '--submission', type=str,  default='sample_submission.json',
                        help='sample submission file for ActivityNet Captions Challenge.')
    parser.add_argument('--split', type=str, default='validation', help='evaluate which split')
    parser.add_argument('-r', '--references', type=str, nargs='+', default=['data/val_1.json', 'data/val_2.json'],
                        help='reference files with ground truth captions to compare results against. delimited (,) str')
    parser.add_argument('--tious', type=float,  nargs='+', default=[0.5],
                        help='Choose the tIoUs to average over.')
    parser.add_argument('-ppv', '--max-proposals-per-video', type=int, default=15,
                        help='maximum propoasls per video.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print intermediate steps.')
    args = parser.parse_args()

    main(args)
