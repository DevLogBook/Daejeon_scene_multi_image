"""Public package surface for the standalone GRM-Matcher.

The model class is added in the matcher implementation stage.  Keeping this
package independent prevents imports from reaching into the existing project.
"""

from .candidate_reranker import (
    GRMCandidateRerankContext,
    GRMCandidateRerankPlugin,
    GRMCandidateRerankResult,
)
from .config import (
    GRM_BUILTIN_CANDIDATE_RERANK_MODES,
    GRMCandidateRerankMode,
    GRMMatcherConfig,
)
from .matcher import GRMMatcher
from .homography_pair_dataset import (
    GRMHomographyPairDataset,
    GRMResolutionBucketBatchSampler,
)
from .real_pair_dataset import GRMRealPairDataset
from .stage25 import GRMStage25Config
from .outputs import (
    GRMMatcherAuxiliary,
    GRMMatcherOutput,
    select_global_matches,
    select_local_matches,
)

__all__ = [
    "GRMMatcher",
    "GRM_BUILTIN_CANDIDATE_RERANK_MODES",
    "GRMCandidateRerankContext",
    "GRMCandidateRerankMode",
    "GRMCandidateRerankPlugin",
    "GRMCandidateRerankResult",
    "GRMMatcherAuxiliary",
    "GRMMatcherConfig",
    "GRMMatcherOutput",
    "GRMHomographyPairDataset",
    "GRMRealPairDataset",
    "GRMResolutionBucketBatchSampler",
    "GRMStage25Config",
    "select_global_matches",
    "select_local_matches",
]
