"""
Database schema for Prime Zeta Research.

This module defines the SQLAlchemy ORM models for storing experiment
data, prime numbers, and analysis results.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    BigInteger,
    REAL,
    String,
    Text,
    Boolean,
    ForeignKey,
    DateTime,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy import event

from ..config import DB_CONFIG

Base = declarative_base()


class Experiment(Base):
    """
    Model for storing experiment parameters and metadata.
    
    Each experiment is defined by unique combination of max_n and c_value.
    """
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True)
    max_n = Column(BigInteger, nullable=False, index=True)  # up to 5e9+
    c_value = Column(REAL, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String(20), default="pending", nullable=False)
    description = Column(Text, nullable=True)
    
    # Relationships
    primes_metadata = relationship("PrimesMetadata", back_populates="experiment")
    results = relationship("Result", back_populates="experiment")
    zeta_matches = relationship("ZetaMatch", back_populates="experiment")
    
    __table_args__ = (
        UniqueConstraint("max_n", "c_value", name="uq_max_n_c_value"),
        Index("idx_experiment_status", "status"),
    )


class PrimesMetadata(Base):
    """
    Model for storing metadata about generated prime numbers.
    
    The actual prime values are stored in binary files, with this table
    keeping track of file locations and statistics.
    """
    __tablename__ = "primes_metadata"
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)
    max_n = Column(BigInteger, nullable=False)
    prime_count = Column(Integer, nullable=False)
    data_file = Column(String(500), nullable=False)
    is_prime_file = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    generation_time_seconds = Column(REAL, nullable=True)
    
    # Relationship
    experiment = relationship("Experiment", back_populates="primes_metadata")
    
    __table_args__ = (
        Index("idx_primes_max_n", "max_n"),
    )


class Result(Base):
    """
    Model for storing computed Need(p), CN(p), Li(p), and R(p) values.
    
    This table stores the main computational results for each prime.
    Due to the large number of entries, consider using partitioned tables
    or separate tables per experiment for very large datasets.
    """
    __tablename__ = "results"
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    prime_value = Column(BigInteger, nullable=False, index=True)  # primes up to 5e9
    need_value = Column(REAL, nullable=True)
    cn_value = Column(REAL, nullable=True)
    li_value = Column(REAL, nullable=True)  # Added: Logarithmic integral Li(p)
    r_residual = Column(REAL, nullable=True)
    
    # Relationship
    experiment = relationship("Experiment", back_populates="results")
    
    __table_args__ = (
        Index("idx_results_experiment_prime", "experiment_id", "prime_value"),
        Index("idx_results_cn", "cn_value"),
        Index("idx_results_r", "r_residual"),
        Index("idx_results_li", "li_value"),
    )


class Spectrum(Base):
    """
    Model for storing FFT spectrum data.
    
    Stores spectrum information for each experiment and C value.
    """
    __tablename__ = "spectra"
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    c_value = Column(REAL, nullable=False)
    frequencies_file = Column(String(500), nullable=False)
    amplitudes_file = Column(String(500), nullable=False)
    peak_frequencies_file = Column(String(500), nullable=True)
    peak_amplitudes_file = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index("idx_spectrum_experiment_c", "experiment_id", "c_value"),
    )


class ZetaMatch(Base):
    """
    Model for storing results of matching spectrum peaks with zeta zeros.
    
    This is the main result table showing the correlation between the
    computed spectrum and theoretical zeta function zeros.
    """
    __tablename__ = "zeta_matches"
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    gamma_index = Column(Integer, nullable=False)
    gamma_value = Column(REAL, nullable=False)
    f_theory = Column(REAL, nullable=False)
    f_peak = Column(REAL, nullable=False)
    relative_error = Column(REAL, nullable=False)
    amplitude = Column(REAL, nullable=True)
    is_good_match = Column(Boolean, default=False)
    
    # Relationship
    experiment = relationship("Experiment", back_populates="zeta_matches")
    
    __table_args__ = (
        Index("idx_zeta_experiment_gamma", "experiment_id", "gamma_index"),
        Index("idx_zeta_error", "relative_error"),
    )


class ZetaZerosCache(Base):
    """
    Cache for zeta function zeros data.
    
    Stores the theoretical gamma_k values for matching.
    """
    __tablename__ = "zeta_zeros_cache"
    
    id = Column(Integer, primary_key=True)
    gamma_index = Column(Integer, nullable=False, unique=True)
    gamma_value = Column(REAL, nullable=False)
    frequency = Column(REAL, nullable=False)  # gamma_k / (2*pi)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index("idx_zeta_zeros_gamma", "gamma_value"),
    )


# Database connection helpers
def create_database(db_path: str, echo: bool = False):
    """
    Create a new database engine and return it.

    Args:
        db_path: Path to SQLite file OR full connection string
                 for PostgreSQL (postgresql+psycopg2://...)
        echo: Whether to log SQL queries

    Returns:
        SQLAlchemy engine
    """
    # PostgreSQL support: if string already looks like URL
    if db_path.startswith("postgresql"):
        engine = create_engine(
            db_path,
            echo=echo,
            pool_size=DB_CONFIG.pool_size,
            max_overflow=DB_CONFIG.max_overflow,
            future=True,
        )
        Base.metadata.create_all(engine)
        return engine

    # Otherwise assume it's a path to SQLite file
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=echo,
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,
            "timeout": 300,  # Increased timeout for large operations
        },
    )

    # SQLite performance tuning
    # Executed on first connection
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        cursor.execute("PRAGMA synchronous=NORMAL")  # Balance of speed and safety
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        cursor.execute("PRAGMA temp_store=MEMORY")  # Temp tables in memory
        cursor.execute("PRAGMA mmap_size=268435456")  # 256MB for memory-mapped I/O
        cursor.close()
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    """
    Create a new database session.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        Session object
    """
    Session = sessionmaker(bind=engine)
    return Session()


if __name__ == "__main__":
    # Test database creation
    import os
    test_db = "test_research.db"
    
    if os.path.exists(test_db):
        os.remove(test_db)
    
    engine = create_database(test_db, echo=False)
    session = get_session(engine)
    
    # Create a test experiment
    experiment = Experiment(
        max_n=1000,
        c_value=10.0,
        status="testing",
        description="Test experiment"
    )
    session.add(experiment)
    session.commit()
    
    print(f"Created experiment with ID: {experiment.id}")
    
    # Clean up
    session.close()
    os.remove(test_db)
    print("Test completed successfully!")
