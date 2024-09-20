import os
import datasets
from functools import partial
from typing import Optional, Dict, Any
from transformers import Trainer
from transformers.models.swin.modeling_swin import SwinConfig, SwinForImageClassification
from transformers.models.vit.feature_extraction_vit import ViTImageProcessor
from transformers import logging

from utils import collate_fn, compute_metrics, create_transform
from parser import parse_train_args

logging.set_verbosity_info()

def launch(args: Optional[Dict[str, Any]] = None):
    feature_extractor_args, dataset_args, model_args, training_args = parse_train_args(args)
    
    ## fetching the feature extractor
    feature_extractor = ViTImageProcessor(do_normalize=feature_extractor_args.do_normalize,
                                          do_rescale=feature_extractor_args.do_rescale,
                                          do_resize=feature_extractor_args.do_resize,
                                          image_mean=feature_extractor_args.image_mean,
                                          image_std=feature_extractor_args.image_std,
                                          resample=feature_extractor_args.resample,
                                          size=feature_extractor_args.size)
    
    # # loading datasets
    ds = datasets.load_dataset(os.path.join(dataset_args.data_dir, "imagenet-1k.py"), trust_remote_code=True)

    # # applying transform
    transform_func = create_transform(feature_extractor)
    prepared_ds = ds.with_transform(transform_func)
    labels = ds['train'].features["label"].names
    
    # # initialzing the model
    config = SwinConfig(embed_dim=model_args.embed_dim,
                        depths=model_args.depths,
                        num_heads=model_args.num_heads,
                        path_norm=model_args.path_norm,
                        num_labels=len(labels),
                        id2label={str(i): c for i, c in enumerate(labels)},
                        label2id={c: str(i) for i, c in enumerate(labels)},
                        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
                        )
    model = SwinForImageClassification(config)
        
    # Instantiate the Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["validation"],
        tokenizer=feature_extractor,
    )

    # Train and save results
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # Evaluate on validation set
    metrics = trainer.evaluate(prepared_ds['validation'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    

if __name__ == "__main__":
    launch()