"""Additional V3 exact-label robustness stage; reuses monitoring and resume contracts."""
import sys
import inspect
from .train_scratch_transfer import main as train


def check_training_interfaces():
    """Fail before loading data/weights when only part of the update was copied."""
    from .scratch_transfer import ScratchTransferObjective
    from . import training_v3, robust_synthetic
    missing=[]
    if 'exact_dense' not in inspect.signature(ScratchTransferObjective.__init__).parameters:
        missing.append('scratch_transfer.py: ScratchTransferObjective.__init__(..., exact_dense=False)')
    if not hasattr(training_v3,'dense_direction_labels'):
        missing.append('training_v3.py: dense_direction_labels')
    if 'severity_fractions' not in robust_synthetic.RobustAugmentConfig.__dataclass_fields__:
        missing.append('robust_synthetic.py: five-tier severity_fractions')
    if missing:
        raise RuntimeError('Incomplete V3 robustness code update. Sync the entire supplied grm_matcher '
                           'update, including scratch_transfer.py, training_v3.py, robust_synthetic.py '
                           'and train_scratch_transfer.py. Missing: '+ '; '.join(missing) +
                           '. Loaded objective from: '+inspect.getfile(ScratchTransferObjective))


def main(argv=None):
    check_training_interfaces()
    argv = sys.argv[1:] if argv is None else argv
    train(['--stage','agri','--synthetic-robustness','--require-architecture','v3', *argv])


if __name__ == '__main__':
    main()
