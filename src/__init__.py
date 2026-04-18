"""Package: hcp-segmentation-engine"""

from src.rfm_scorer import (  # noqa: F401
    compute_rfm_scores,
    get_top_hcps,
    summarise_rfm_segments,
)
from src.calling_plan_allocator import (  # noqa: F401
    allocate_calls,
    calculate_priority_score,
    summarise_allocation,
)
