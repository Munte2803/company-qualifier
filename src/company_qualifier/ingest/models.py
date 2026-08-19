from pydantic import BaseModel


class Address(BaseModel):
       country_code: str
       region_name: str | None=None
       town: str | None=None
       latitude: float | None=None
       longitude: float | None=None   

class Naics(BaseModel):
       code: str
       label: str
       share: float | None=None     

class Company(BaseModel):
       row_index: int
       operational_name: str|None=None
       address: Address
       description: str
       business_model: list[str]
       employee_count: int|None=None
       revenue: float|None=None
       is_public: bool
       year_founded: int|None=None
       website: str|None=None
       primary_naics: Naics
       secondary_naics: Naics|None=None
       target_markets: list[str]
       core_offerings: list[str]


