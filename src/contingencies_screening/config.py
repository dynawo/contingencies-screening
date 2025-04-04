from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration settings for the contingency screening application.

    Attributes:
        HADES_FOLDER: Name of the Hades output folder.
        DYNAWO_FOLDER: Name of the Dynawo output folder.
        REPLAY_NUM: Number of top contingencies to replay.
        DEFAULT_SCORE: Default score assigned to contingencies.
        N_THREADS_LAUNCHER: Number of threads for parallel contingency execution within a snapshot.
        N_THREADS_SNAPSHOT: Number of threads for parallel snapshot execution.
    """

    HADES_FOLDER: str = "hades"
    DYNAWO_FOLDER: str = "dynawo"
    REPLAY_NUM: int = 25
    DEFAULT_SCORE: int = 1
    N_THREADS_LAUNCHER: int = 1
    N_THREADS_SNAPSHOT: int = 4

    model_config = SettingsConfigDict(env_prefix="CONT_SCR")  # Optional: Prefix for env vars


settings = Settings()
