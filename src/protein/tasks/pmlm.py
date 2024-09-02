import logging
import os

from fairseq import utils
from fairseq.data import Dictionary

from fairseq.tasks import FairseqTask, register_task
logger = logging.getLogger(__name__)

@register_task('prot_pmlm')
class ProtPairwiseMaskedLMTask(FairseqTask):

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.args = args
        self.dictionary = dictionary
        if not hasattr(args, 'max_positions'):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions
        # add mask token
        self.mask_idx = dictionary.add_symbol("<mask>")

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    @property
    def source_dictionary(self):
        return self.dictionary