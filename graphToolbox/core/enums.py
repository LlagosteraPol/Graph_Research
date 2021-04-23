__author__ = 'Pol Llagostera Blasco'

from enum import Enum


class GraphType(Enum):
    Tree = 1
    PureCycle = 2
    OrderedCycles = 3
    Others = 4


class FormatType(Enum):
    CSV = 1
    JSON = 2
    HTML = 3
    Excel = 4
    SQL = 5
