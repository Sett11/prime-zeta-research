"""
Config loader module.

Loads configuration from YAML file and provides access to parameters.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from loguru import logger


class ConfigLoader:
    """
    Configuration loader from YAML file.

    Usage example:
        config = ConfigLoader()
        max_n = config.get("experiment", "max_n")
        c_value = config.get_experiment_param("c_value")
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to config.yaml file. If None, searches in current directory.
        """
        if config_path is None:
            # Search for config.yaml in current and parent directories
            self.config_path = self._find_config_file()
        else:
            self.config_path = Path(config_path)

        self._config: Optional[Dict[str, Any]] = None

        if self.config_path and self.config_path.exists():
            self._load_config()
        else:
            logger.warning(f"Config file not found: {self.config_path}")

    def _find_config_file(self) -> Optional[Path]:
        """Search for config.yaml in current and parent directories."""
        current = Path.cwd()

        for _ in range(5):  # Check up to 5 levels up
            config_file = current / "config.yaml"
            if config_file.exists():
                return config_file
            current = current.parent

        return None

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)

            logger.info(f"Config loaded from: {self.config_path}")
            logger.info(f"Experiment parameters: max_n={self.get('experiment', 'max_n')}, "
                       f"c_value={self.get('experiment', 'c_value')}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._config = {}

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get parameter value from configuration.

        Args:
            section: Configuration section (experiment, database, etc.)
            key: Parameter key
            default: Default value

        Returns:
            Parameter value or default
        """
        if self._config is None:
            return default

        section_data = self._config.get(section, {})
        return section_data.get(key, default)

    def get_experiment_param(self, key: str, default: Any = None) -> Any:
        """
        Get experiment parameter.

        Args:
            key: Parameter key
            default: Default value

        Returns:
            Parameter value or default
        """
        return self.get("experiment", key, default)

    def get_db_param(self, key: str, default: Any = None) -> Any:
        """
        Get database parameter.

        Args:
            key: Parameter key
            default: Default value

        Returns:
            Parameter value or default
        """
        return self.get("database", key, default)

    def get_analysis_param(self, key: str, default: Any = None) -> Any:
        """
        Get analysis parameter.

        Args:
            key: Parameter key
            default: Default value

        Returns:
            Parameter value or default
        """
        return self.get("analysis", key, default)

    def get_output_param(self, key: str, default: Any = None) -> Any:
        """
        Get output parameter.

        Args:
            key: Parameter key
            default: Default value

        Returns:
            Parameter value or default
        """
        return self.get("output", key, default)

    def get_table_name(self, table_type: str) -> str:
        """
        Get database table name.

        Args:
            table_type: Table type (primes, need_cn, experiments)

        Returns:
            Table name
        """
        tables = self.get("database", "tables", {})
        return tables.get(table_type, table_type)

    # -------- Experiment/Analysis parameters (convenience getters) --------

    @property
    def max_n(self) -> int:
        """Returns max_n from configuration."""
        return self.get_experiment_param("max_n", 10_000_000)

    @property
    def c_value(self) -> float:
        """Returns c_value from configuration."""
        return self.get_experiment_param("c_value", 10.0)

    @property
    def db_path(self) -> str:
        """
        Returns database connection string.

        For SQLite this is a file path (research.db).
        For PostgreSQL - full connection string like
        postgresql+psycopg2://user:password@host:port/db.
        """
        engine = self.get("database", "engine", "sqlite")

        if engine == "postgres":
            pg_cfg = self.get("database", "postgres", {}) or {}
            user = pg_cfg.get("user", "prime_zeta")
            password = pg_cfg.get("password", "prime_zeta")
            host = pg_cfg.get("host", "localhost")
            port = pg_cfg.get("port", 5432)
            db = pg_cfg.get("db", "prime_zeta")
            return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

        # Default - SQLite file
        return self.get_db_param("path", "research.db")

    @property
    def gamma_k_file(self) -> str:
        """Returns path to gamma_k file."""
        return self.get_experiment_param("gamma_k_file", "data/gamma_k_5000000.txt")

    @property
    def results_dir(self) -> str:
        """Returns results directory."""
        return self.get_output_param("directory", "results")

    def get_primes_file(self) -> str:
        """Path to binary primes file: data/primes/primes_{max_n}.npy."""
        if self.config_path is not None:
            root = Path(self.config_path).parent
        else:
            root = Path.cwd()
        return str(root / "data" / "primes" / f"primes_{self.max_n}.npy")

    @property
    def output_format(self) -> str:
        """Returns output format."""
        return self.get_output_param("format", "json+png")

    @property
    def log_level(self) -> str:
        """Returns logging level."""
        return self.get("logging", "level", "INFO")

    # -------- Wavelet analysis / scalogram parameters --------

    def get_wavelet_param(self, key: str, default: Any = None) -> Any:
        """
        Get parameter from analysis.wavelet section.

        Args:
            key: Parameter key within wavelet block
            default: Default value
        """
        # In config.yaml, wavelet block is inside analysis
        wavelet_cfg = self.get("analysis", "wavelet", {})
        if isinstance(wavelet_cfg, dict):
            return wavelet_cfg.get(key, default)
        return default

    @property
    def wavelet_scalogram_enabled(self) -> bool:
        """Whether automatic scalogram generation is enabled."""
        return bool(self.get_wavelet_param("scalogram_enabled", False))

    @property
    def wavelet_scalogram_max_freq(self) -> Optional[float]:
        """
        Maximum frequency for scalogram display.
        None means use full available frequency range.
        """
        value = self.get_wavelet_param("scalogram_max_freq", None)
        return value

    @property
    def wavelet_scalogram_filename(self) -> str:
        """Filename for saving scalogram in results directory."""
        return str(self.get_wavelet_param("scalogram_filename", "scalogram.png"))

    def reload(self):
        """Reload configuration from file."""
        self._load_config()

    def __repr__(self) -> str:
        if self._config:
            return f"ConfigLoader(config={self.config_path}, max_n={self.max_n}, c_value={self.c_value})"
        return "ConfigLoader(not loaded)"


# Global instance
_global_config: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Returns global ConfigLoader instance.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigLoader instance
    """
    global _global_config

    if _global_config is None:
        _global_config = ConfigLoader(config_path)

    return _global_config


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration and return dictionary.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config = get_config(config_path)
    return config._config if config._config else {}


if __name__ == "__main__":
    # Test configuration loading
    config = ConfigLoader()

    if config._config:
        print(f"Config loaded: {config.config_path}")
        print(f"max_n: {config.max_n}")
        print(f"c_value: {config.c_value}")
        print(f"db_path: {config.db_path}")
        print(f"primes table: {config.get_table_name('primes')}")
    else:
        print("Config not found!")
