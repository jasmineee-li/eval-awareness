from pydantic import BaseModel, field_validator
from typing import Optional
import sys
from math import inf

class NumericalRange(BaseModel):
    lower_bound: Optional[float]
    upper_bound: Optional[float]
