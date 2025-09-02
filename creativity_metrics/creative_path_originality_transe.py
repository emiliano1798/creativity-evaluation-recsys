# creativity_metrics/creative_path_originality_transe.py

from creativity_metrics.creative_path_originality_base import CreativePathOriginalityBase

class CreativePathOriginalityTransE(CreativePathOriginalityBase):
    def __init__(self, dataset_name='ml1m'):
        super().__init__(dataset_name=dataset_name, model_name='transe')

