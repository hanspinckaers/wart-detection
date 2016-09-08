import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from analyze_training import cross_validate_with_participants

print cross_validate_with_participants
