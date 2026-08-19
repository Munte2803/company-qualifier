from pydantic import BaseModel


class NumericFilter(BaseModel):
    field: str                   
    op: str                  
    value: float                 


class QueryPlan(BaseModel):
    country_codes: list[str] = []      
    numeric_filters: list[NumericFilter] = []
    semantic_text: str               
    hyde_descriptions: list[str] = [] 
    unsupported: list[str] = []  
    boolean_filters: dict[str, bool] = {}

class Verdict(BaseModel):
    row_index: int
    verdict: str                 
    evidence: str                 
    reason: str                  


class JudgeResponse(BaseModel):
    verdicts: list[Verdict]