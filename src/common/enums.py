from enum import StrEnum


class Scenario(StrEnum):
    NONE = "none"
    SLOWDOWN = "slowdown"
    STOP = "stop"


class FusionRule(StrEnum):
    WEIGHTED_MEAN = "weighted_mean"
    MEDIAN = "median"
    MAX = "max"


class DetectorMode(StrEnum):
    LEVEL = "level"
    SLOPE = "slope"
    BOTH = "both"


class LostRule(StrEnum):
    ANY = "any"
    NOT_DEAF = "not_deaf"
    NOT_DEAF_EPISODE = "not_deaf_episode"


class RobustMode(StrEnum):
    NONE = "none"
    INFLATE = "inflate"
    NIS_GATE = "nis_gate"


class Reference(StrEnum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class ShadowingMode(StrEnum):
    FIELD = "field"
    INDEPENDENT = "independent"


class ReceptionModel(StrEnum):
    CURVE = "curve"
    THRESHOLD = "threshold"


class BackgroundModel(StrEnum):
    INDEPENDENT = "independent"
    BURSTY = "bursty"


class BurstScope(StrEnum):
    RECEIVER = "receiver"
    LINK = "link"


class Strategy(StrEnum):
    DIRECT = "direct"
    BEST_RSSI = "best_rssi"
    DQREM = "dqrem"
    DQREM_NO_SLOPE = "dqrem_no_slope"
