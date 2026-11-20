from functools import cache
from knowledge.service.import_file_service import ImportFileService
from knowledge.service.query_service import QueryService


@cache
def get_import_file_service() -> ImportFileService:
    return ImportFileService()

@cache
def get_query_service() -> QueryService:
    return QueryService()