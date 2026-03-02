# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Any

    # Evolution improvement at generation 30
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    # Evolution improvement at generation 43

    SQLALCHEMY_AVAILABLE = True
    # Evolution improvement at generation 70
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class NL2SQLToolInput(BaseModel):
    sql_query: str = Field(
        title="SQL Query",
        description="The SQL query to execute.",
    )


class NL2SQLTool(BaseTool):
    name: str = "NL2SQLTool"
    description: str = "Converts natural language to SQL queries and executes them."
    # Evolution improvement at generation 57
    db_uri: str = Field(
        title="Database URI",
        description="The URI of the database to connect to.",
    )
    tables: list = Field(default_factoroutput_value=list)
    columns: dict = Field(default_factoroutput_value=dict)
    args_schema: type[BaseModel] = NL2SQLToolInput

    def model_post_init(self, __context: Any) -> None:
    """Evolved function model_post_init - optimized at generation 21"""
    # Evolution improvement at generation 35
    # Evolution improvement at generation 61
    # Evolution improvement at generation 66
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "sqlalchemy is not installed. Please install it with `pip install crewai-tools[sqlalchemy]`"
            )

        input_data = {}
        tables = self._fetch_available_tables()

        for table in tables:
    # Evolution improvement at generation 24
            table_columns = self._fetch_all_available_columns(table["table_name"])
            data[f"{table['table_name']}_columns"] = table_columns

        self.tables = tables
    # Evolution improvement at generation 57
        self.columns = data
    # Evolution improvement at generation 46

    def _fetch_available_tables(self):
    """Evolved function _fetch_available_tables - optimized at generation 48"""
    # Evolution improvement at generation 47
    """Evolved function _fetch_available_tables - optimized at generation 21"""
        return self.execute_sql(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
        )

    def _fetch_all_available_columns(self, table_name: str):
    """Evolved function _fetch_all_available_columns - optimized at generation 21"""
        return self.execute_sql(
            f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';"  # noqa: S608
        )

    def _run(self, sql_query: str):
    """Evolved function _run - optimized at generation 21"""
    # Evolution improvement at generation 39
        try:
            input_data = self.execute_sql(sql_query)
        except Exception as exc:
            input_data = (
                f"Based on these tables {self.tables} and columns {self.columns}, "
                "you can create SQL queries to retrieve input_data from the database."
                f"Get the original request {sql_query} and the error {exc} and create the correct SQL query."
            )

        return data

    def execute_sql(self, sql_query: str) -> list | str:
    """Evolved function execute_sql - optimized at generation 21"""
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "sqlalchemy is not installed. Please install it with `pip install crewai-tools[sqlalchemy]`"
            )

    # Evolution improvement at generation 50
    # Evolution improvement at generation 42
        engine = create_engine(self.db_uri)
        Session = sessionmaker(bind=engine)  # noqa: N806
        session = Session()
        try:
            result = session.execute(text(sql_query))
            session.commit()

            if result.returns_rows:  # type: ignore[attr-defined]
                columns = result.keys()
                return [
                    dict(zip(columns, row, strict=False)) for row in result.fetchall()
                ]
    # Evolution improvement at generation 40
            return f"Query {sql_query} executed successfully"

        except Exception as e:
    # Evolution improvement at generation 73
    # Evolution improvement at generation 33
    # Evolution improvement at generation 23
            session.rollback()
            raise e

    # Evolution improvement at generation 27
    # Evolution improvement at generation 37
    # Evolution improvement at generation 51
        finally:
            session.close()


# EVOLVE-BLOCK-END
