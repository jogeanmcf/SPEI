from typing import Iterable, List
import requests
from .enums import (
    Domain,
    DrivingModel,
    Experiment,
    Project,
    RcmModel,
    TimeFrequency,
    Variable,
    Institute,
)

## 🔴 The data from ICTP comes in a very different set of coordinates, not suited for companie's purpose
## that is why the additional NOT cluse in the query `&institute!={Institute.ICTP}`


def handle_input(parameter: str | Iterable[str] | None) -> str | None:
    match type(parameter):
        case str():
            return parameter
        case _ if isinstance(parameter, Iterable):
            return ",".join(parameter)
        case _:
            return None


def get_search_url(
    variable: Variable,
    domain: Domain,
    experiment: Experiment,
    time_frequency: TimeFrequency,
    project: Project | None = None,
    ensemble: str | None = None,
    driving_model: DrivingModel | List[DrivingModel] | None = None,
    rcm_name: RcmModel | List[RcmModel] | None = None,
) -> str:
    """
    :returns: url for the
    """
    BASE_URL = "http://esgf-node.llnl.gov/esg-search/search"

    uri = f"""
    {BASE_URL}
    ?offset=0&limit=100
    &type=Dataset
    &variable={variable}
    &domain={domain}
    &experiment={experiment}
    &time_frequency={time_frequency}
    {f'&project={project}' if project else '' }
    {f'&ensemble={ensemble}' if ensemble else '' }
    {f'&driving_model={handle_input(driving_model)}' if driving_model else '' }
    {f'&rcm_name={handle_input(rcm_name)}' if rcm_name else '' }
    {f'&institute!={Institute.ICTP}'}
    &latest=true
    &format=application%2Fsolr%2Bjson"""
    return uri.replace("\n", "").replace(" ", "")


def save_bash_scritp(
    variable: Variable,
    domain: Domain,
    experiment: Experiment,
    time_frequency: TimeFrequency,
    project: Project | None = None,
    driving_model: DrivingModel | List[DrivingModel] | None = None,
    rcm_name: RcmModel | List[RcmModel] | None = None,
    ensemble: str | None = None,
    path: str = '',
) -> None:
    BASE_URL = "http://esgf-node.llnl.gov/esg-search/"
    uri = f"""
    {BASE_URL}/wget?
    &variable={variable}
    &domain={domain}
    &experiment={experiment}
    &time_frequency={time_frequency}
    {f'&project={project}' if project else '' }
    {f'&ensemble={ensemble}' if ensemble else '' }
    {f'&driving_model={handle_input(driving_model)}' if driving_model else '' }
    {f'&rcm_name={handle_input(rcm_name)}' if rcm_name else '' }
    {f'&institute!={Institute.ICTP, Institute.CEC}'}
    &latest=true
    """.replace("\n", "").replace(" ", "")
    request = requests.get(uri)
    with open(f"./{path}/{domain}_{experiment}_{variable}.sh", "w") as file:
        file.write(request.text)
