from dataclasses import dataclass

@dataclass
class AGENT_STATE:
    EXECUTING_TRAJ: str = "EXECUTING_TRAJ"
    SETTLING: str = "SETTLING"
    UNKNOWN: str = "UNKNOWN"