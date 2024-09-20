import torch
import evaluate
import numpy as np

metric = evaluate.load("accuracy.py")

def create_transform(feature_extractor):
    def transform(example_batch): 
        # Take a list of PIL images and turn them to pixel values
        inputs = feature_extractor([x.convert('RGB') for x in example_batch['image']], return_tensors='pt')
        inputs['label'] = example_batch['label']
        return inputs
    return transform


def collate_fn(batch):
  #data collator
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

def compute_metrics(p):
  # function which calculates accuracy for a certain set of predictions
  return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)