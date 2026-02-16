"""
Database operations for Prime Zeta Research.

This module provides CRUD operations and high-level database managers
for storing and retrieving research data.
"""

import os
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from loguru import logger

from .schema import (
    Experiment,
    PrimesMetadata,
    Result,
    Spectrum,
    ZetaMatch,
    ZetaZerosCache,
    create_database,
    get_session,
)
from ..config import ResearchConfig


class ExperimentManager:
    """Manager for experiment records in the database."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_or_create(
        self, 
        max_n: int, 
        c_value: float,
        description: Optional[str] = None
    ) -> Experiment:
        """
        Get existing experiment or create new one.
        
        Args:
            max_n: Maximum number for prime generation
            c_value: Window scaling parameter
            description: Optional description
            
        Returns:
            Experiment instance
        """
        # Robust get/create experiment pattern:
        # 1) Try to find existing record by (max_n, c_value)
        # 2) If not found - create Experiment and try to commit
        # 3) On IntegrityError (duplicate) do rollback and re-read
        # 1) Try to find existing experiment
        experiment = (
            self.session.query(Experiment)
            .filter(Experiment.max_n == max_n, Experiment.c_value == c_value)
            .first()
        )
        if experiment is not None:
            return experiment

        # 2) Try to create new one
        experiment = Experiment(
            max_n=max_n,
            c_value=c_value,
            description=description,
            status="pending",
        )
        self.session.add(experiment)
        try:
            self.session.commit()
        except IntegrityError:
            # 3) In race conditions or re-runs:
            # rollback and re-read the existing experiment.
            self.session.rollback()
            logger.warning(
                f"IntegrityError when creating Experiment(max_n={max_n}, c={c_value}), "
                f"trying to re-read existing record."
            )

            # Refresh session to see changes from other transactions
            self.session.expire_all()

            # Try to find existing experiment after rollback
            experiment = (
                self.session.query(Experiment)
                .filter(Experiment.max_n == max_n, Experiment.c_value == c_value)
                .first()
            )
            
            if experiment is None:
                # If still not found, try UPSERT for PostgreSQL
                engine = self.session.get_bind()
                dialect = engine.dialect.name
                if dialect == "postgresql":
                    if description is None:
                        description = f"Experiment max_n={max_n}, c={c_value}"

                    # Use ON CONFLICT DO UPDATE SET id = id RETURNING * 
                    # to guaranteed get a record (existing or new)
                    result = self.session.execute(
                        text(
                            """
                            INSERT INTO experiments (max_n, c_value, created_at, status, description)
                            VALUES (:max_n, :c_value, NOW(), :status, :description)
                            ON CONFLICT (max_n, c_value) 
                            DO UPDATE SET id = experiments.id
                            RETURNING id, max_n, c_value, created_at, status, description
                            """
                        ),
                        {
                            "max_n": max_n,
                            "c_value": c_value,
                            "status": "pending",
                            "description": description,
                        },
                    )
                    self.session.commit()
                    
                    # Get record from result or do final SELECT
                    row = result.fetchone()
                    if row:
                        # Load object by ID from result
                        experiment = self.session.get(Experiment, row[0])
                    else:
                        # If RETURNING returned nothing (shouldn't happen), do SELECT
                        experiment = (
                            self.session.query(Experiment)
                            .filter(Experiment.max_n == max_n, Experiment.c_value == c_value)
                            .first()
                        )
                else:
                    # For SQLite just do final SELECT
                    experiment = (
                        self.session.query(Experiment)
                        .filter(Experiment.max_n == max_n, Experiment.c_value == c_value)
                        .first()
                    )

            if experiment is None:
                logger.error(
                    f"Failed to create/read Experiment(max_n={max_n}, c={c_value}) "
                    f"after IntegrityError and all recovery attempts."
                )
                raise RuntimeError(
                    f"Critical error: failed to get Experiment(max_n={max_n}, c={c_value}) "
                    f"after IntegrityError. Possible database or transaction issue."
                )

        return experiment
    
    def update_status(self, experiment: Experiment, status: str):
        """Update experiment status."""
        if experiment is None:
            from loguru import logger
            logger.warning(f"update_status: experiment is None, status '{status}' skipped")
            return
        experiment.status = status
        self.session.commit()
    
    def get_by_id(self, experiment_id: int) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self.session.query(Experiment).filter(
            Experiment.id == experiment_id
        ).first()
    
    def get_all(self, status: Optional[str] = None) -> List[Experiment]:
        """Get all experiments, optionally filtered by status."""
        query = self.session.query(Experiment)
        if status:
            query = query.filter(Experiment.status == status)
        return query.order_by(Experiment.created_at.desc()).all()

    def get_by_params(self, max_n: int, c_value: float) -> Optional[Experiment]:
        """
        Get experiment by (max_n, c_value) pair without creating new one.

        IMPORTANT: c_value is stored in DB as real number (REAL/DOUBLE),
        so exact equality comparison (==) may fail due to
        rounding errors. Instead we use a small window around c_value.
        """
        # Small tolerance for c_value to account for rounding errors
        eps = 1e-6
        query = self.session.query(Experiment).filter(
            Experiment.max_n == max_n,
            Experiment.c_value >= c_value - eps,
            Experiment.c_value <= c_value + eps,
        )
        # If for some reason there are multiple experiments in this range,
        # take the first (they should be unique by (max_n, c_value) anyway).
        return query.order_by(Experiment.id.asc()).first()


class PrimesMetadataManager:
    """Manager for prime number metadata."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def save(
        self,
        experiment: Experiment,
        max_n: int,
        prime_count: int,
        data_file: str,
        generation_time: float,
        is_prime_file: Optional[str] = None
    ) -> PrimesMetadata:
        """Save primes metadata to database."""
        metadata = PrimesMetadata(
            experiment_id=experiment.id,
            max_n=max_n,
            prime_count=prime_count,
            data_file=data_file,
            is_prime_file=is_prime_file,
            generation_time_seconds=generation_time
        )
        self.session.add(metadata)
        self.session.commit()
        return metadata
    
    def get_by_max_n(self, max_n: int) -> Optional[PrimesMetadata]:
        """Get metadata by max_n value."""
        return self.session.query(PrimesMetadata).filter(
            PrimesMetadata.max_n == max_n
        ).first()


class ResultManager:
    """Manager for computation results (Need, CN, Li, R values)."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def save_batch(
        self,
        experiment: Experiment,
        primes: np.ndarray,
        need_values: Optional[np.ndarray] = None,
        cn_values: Optional[np.ndarray] = None,
        li_values: Optional[np.ndarray] = None,
        r_values: Optional[np.ndarray] = None,
        batch_size: int = 50000
    ):
        """
        Save results in batches to avoid memory issues.
        Uses bulk_insert_mappings for better performance.
        
        Args:
            experiment: Experiment instance
            primes: Array of prime numbers
            need_values: Optional array of Need(p) values
            cn_values: Optional array of CN(p) values
            li_values: Optional array of Li(p) values
            r_values: Optional array of R(p) values
            batch_size: Number of records per batch
        """
        total = len(primes)
        experiment_id = experiment.id
        
        # SQLite optimization for bulk insert (not applied for PostgreSQL)
        engine = self.session.get_bind()
        if engine.dialect.name == "sqlite":
            self.session.execute(text("PRAGMA synchronous = NORMAL"))
            self.session.execute(text("PRAGMA journal_mode = WAL"))
            self.session.execute(text("PRAGMA cache_size = -64000"))  # 64MB cache
            self.session.execute(text("PRAGMA temp_store = MEMORY"))
        
        # Commit every N batches for balance between speed and safety
        commit_interval = 5
        
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            
            # Use bulk_insert_mappings instead of creating ORM objects
            # This is much faster for large data volumes
            batch_data = []
            for j in range(i, end_idx):
                row = {
                    "experiment_id": experiment_id,
                    "prime_value": int(primes[j]),
                }
                if need_values is not None:
                    row["need_value"] = float(need_values[j])
                if cn_values is not None:
                    row["cn_value"] = float(cn_values[j])
                if li_values is not None:
                    row["li_value"] = float(li_values[j])
                if r_values is not None:
                    row["r_residual"] = float(r_values[j])
                batch_data.append(row)
            
            # Bulk insert without creating ORM objects
            self.session.bulk_insert_mappings(Result, batch_data)
            
            # Commit every N batches
            if (i // batch_size + 1) % commit_interval == 0 or end_idx == total:
                self.session.commit()
            
            # Progress
            if (i + batch_size) % (batch_size * 10) == 0 or end_idx == total:
                print(f"  Saved {end_idx}/{total} results ({100*end_idx/total:.1f}%)")
        
        # Final commit just in case
        self.session.commit()
    
    def get_by_experiment(self, experiment_id: int, limit: Optional[int] = None):
        """Get results for an experiment."""
        query = self.session.query(Result).filter(
            Result.experiment_id == experiment_id
        ).order_by(Result.prime_value)
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def get_arrays(self, experiment_id: int, batch_size: int = 1_000_000, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get results as numpy arrays (including Li values) with batched loading.
        
        Uses yield_per() for memory-efficient streaming of large datasets.
        
        Args:
            experiment_id: Experiment ID
            batch_size: Number of records to load per batch
            verbose: Whether to log progress
            
        Returns:
            Tuple of (primes, need_values, cn_values, li_values, r_values)
        """
        from loguru import logger
        from tqdm import tqdm
        
        # Get total count first
        total_count = self.session.query(Result).filter(
            Result.experiment_id == experiment_id
        ).count()
        
        if verbose:
            logger.info(f"Loading {total_count:,} records from database (batches of {batch_size:,})...")
        
        # Query with yield_per for streaming
        query = self.session.query(Result).filter(
            Result.experiment_id == experiment_id
        ).order_by(Result.prime_value).yield_per(batch_size)
        
        # Pre-allocate numpy arrays for better performance
        if verbose:
            logger.info(f"Allocating memory for {total_count:,} records...")
        primes = np.empty(total_count, dtype=np.int64)
        need_values = np.empty(total_count, dtype=np.float64)
        cn_values = np.empty(total_count, dtype=np.float64)
        li_values = np.empty(total_count, dtype=np.float64)
        r_values = np.empty(total_count, dtype=np.float64)
        
        # Progress bar
        pbar = tqdm(total=total_count, desc="Loading data", unit=" records", disable=not verbose) if verbose else None
        
        # Load in batches and fill arrays directly
        loaded = 0
        batch_primes = []
        batch_need = []
        batch_cn = []
        batch_li = []
        batch_r = []
        
        for result in query:
            batch_primes.append(result.prime_value)
            batch_need.append(result.need_value if result.need_value is not None else np.nan)
            batch_cn.append(result.cn_value if result.cn_value is not None else np.nan)
            batch_li.append(result.li_value if result.li_value is not None else np.nan)
            batch_r.append(result.r_residual if result.r_residual is not None else np.nan)
            
            # When batch is full, copy to pre-allocated arrays
            if len(batch_primes) >= batch_size:
                end_idx = loaded + len(batch_primes)
                primes[loaded:end_idx] = batch_primes
                need_values[loaded:end_idx] = batch_need
                cn_values[loaded:end_idx] = batch_cn
                li_values[loaded:end_idx] = batch_li
                r_values[loaded:end_idx] = batch_r
                
                loaded += len(batch_primes)
                
                if pbar:
                    pbar.update(len(batch_primes))
                
                # Log progress every 1M records
                if verbose and loaded % 1_000_000 == 0:
                    logger.info(f"Loaded {loaded:,}/{total_count:,} records ({100*loaded/total_count:.1f}%)")
                
                # Clear batch
                batch_primes = []
                batch_need = []
                batch_cn = []
                batch_li = []
                batch_r = []
        
        # Process remaining items
        if batch_primes:
            end_idx = loaded + len(batch_primes)
            primes[loaded:end_idx] = batch_primes
            need_values[loaded:end_idx] = batch_need
            cn_values[loaded:end_idx] = batch_cn
            li_values[loaded:end_idx] = batch_li
            r_values[loaded:end_idx] = batch_r
            loaded += len(batch_primes)
            if pbar:
                pbar.update(len(batch_primes))
        
        if pbar:
            pbar.close()
        
        if verbose:
            logger.info(f"Loading complete. Data is already in numpy arrays.")
        
        if verbose:
            logger.info(f"Data loaded: {len(primes):,} primes")
        
        return primes, need_values, cn_values, li_values, r_values
    
    def iter_arrays(
        self,
        experiment_id: int,
        batch_size: int = 1_000_000,
        verbose: bool = True,
    ):
        """
        Streaming API: returns data in batches as numpy arrays.

        Instead of collecting all arrays fully in memory,
        this method allows iterating over the DB in blocks:

            for primes, need, cn, li, r in db.results.iter_arrays(...):
                # process batch

        Args:
            experiment_id: Experiment ID
            batch_size: Batch size (number of rows)
            verbose: Log progress

        Yields:
            (primes_batch, need_batch, cn_batch, li_batch, r_batch)
        """
        from loguru import logger
        from tqdm import tqdm

        total_count = self.session.query(Result).filter(
            Result.experiment_id == experiment_id
        ).count()

        if verbose:
            logger.info(
                f"Streaming {total_count:,} records "
                f"(batches of {batch_size:,})..."
            )

        query = (
            self.session.query(Result)
            .filter(Result.experiment_id == experiment_id)
            .order_by(Result.prime_value)
            .yield_per(batch_size)
        )

        pbar = tqdm(
            total=total_count,
            desc="Streaming data",
            unit=" records",
            disable=not verbose,
        )

        batch_primes: list[int] = []
        batch_need: list[float] = []
        batch_cn: list[float] = []
        batch_li: list[float] = []
        batch_r: list[float] = []

        for idx, result in enumerate(query, start=1):
            batch_primes.append(result.prime_value)
            batch_need.append(
                result.need_value if result.need_value is not None else np.nan
            )
            batch_cn.append(
                result.cn_value if result.cn_value is not None else np.nan
            )
            batch_li.append(
                result.li_value if result.li_value is not None else np.nan
            )
            batch_r.append(
                result.r_residual if result.r_residual is not None else np.nan
            )

            if len(batch_primes) >= batch_size:
                yield (
                    np.array(batch_primes, dtype=np.int64),
                    np.array(batch_need, dtype=np.float64),
                    np.array(batch_cn, dtype=np.float64),
                    np.array(batch_li, dtype=np.float64),
                    np.array(batch_r, dtype=np.float64),
                )
                pbar.update(len(batch_primes))
                batch_primes.clear()
                batch_need.clear()
                batch_cn.clear()
                batch_li.clear()
                batch_r.clear()

        # Trailing batch
        if batch_primes:
            yield (
                np.array(batch_primes, dtype=np.int64),
                np.array(batch_need, dtype=np.float64),
                np.array(batch_cn, dtype=np.float64),
                np.array(batch_li, dtype=np.float64),
                np.array(batch_r, dtype=np.float64),
            )
            pbar.update(len(batch_primes))

        pbar.close()
    
    def delete_by_experiment(self, experiment_id: int):
        """Delete all results for an experiment."""
        self.session.query(Result).filter(
            Result.experiment_id == experiment_id
        ).delete()
        self.session.commit()


class SpectrumManager:
    """Manager for FFT spectrum data."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def save(
        self,
        experiment: Experiment,
        c_value: float,
        frequencies_file: str,
        amplitudes_file: str,
        peak_frequencies_file: Optional[str] = None,
        peak_amplitudes_file: Optional[str] = None
    ) -> Spectrum:
        """Save spectrum metadata to database."""
        spectrum = Spectrum(
            experiment_id=experiment.id,
            c_value=c_value,
            frequencies_file=frequencies_file,
            amplitudes_file=amplitudes_file,
            peak_frequencies_file=peak_frequencies_file,
            peak_amplitudes_file=peak_amplitudes_file
        )
        self.session.add(spectrum)
        self.session.commit()
        return spectrum
    
    def get_by_experiment_c(self, experiment_id: int, c_value: float) -> Optional[Spectrum]:
        """Get spectrum by experiment ID and C value."""
        return self.session.query(Spectrum).filter(
            Spectrum.experiment_id == experiment_id,
            Spectrum.c_value == c_value
        ).first()


class ZetaMatchManager:
    """Manager for zeta function zero matching results."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def save_batch(
        self,
        experiment: Experiment,
        matches: List[Dict[str, Any]],
        error_threshold: float = 0.01
    ):
        """
        Save matching results in batch.
        
        Args:
            experiment: Experiment instance
            matches: List of match dictionaries with keys:
                - gamma_index: Index of the zero
                - gamma_value: Value of gamma_k
                - f_theory: Theoretical frequency
                - f_peak: Found peak frequency
                - relative_error: Relative error
                - amplitude: Peak amplitude
            error_threshold: Threshold for marking good matches
        """
        zeta_matches = [
            ZetaMatch(
                experiment_id=experiment.id,
                gamma_index=m["gamma_index"],
                gamma_value=m["gamma_value"],
                f_theory=m["f_theory"],
                f_peak=m["f_peak"],
                relative_error=m["relative_error"],
                amplitude=m.get("amplitude"),
                is_good_match=m["relative_error"] < error_threshold
            )
            for m in matches
        ]
        
        self.session.bulk_save_objects(zeta_matches)
        self.session.commit()
    
    def get_statistics(self, experiment_id: int) -> Dict[str, Any]:
        """Get matching statistics for an experiment."""
        matches = self.session.query(ZetaMatch).filter(
            ZetaMatch.experiment_id == experiment_id
        ).all()
        
        if not matches:
            return {"total": 0}
        
        errors = [m.relative_error for m in matches]
        good_matches = sum(1 for m in matches if m.is_good_match)
        
        return {
            "total": len(matches),
            "good_matches": good_matches,
            "good_percentage": 100 * good_matches / len(matches),
            "mean_error": np.mean(errors),
            "median_error": np.median(errors),
            "min_error": min(errors),
            "max_error": max(errors),
            "error_percentiles": {
                "1%": np.percentile(errors, 1),
                "5%": np.percentile(errors, 5),
                "10%": np.percentile(errors, 10),
                "50%": np.percentile(errors, 50),
                "90%": np.percentile(errors, 90),
            }
        }
    
    def get_best_matches(
        self, 
        experiment_id: int, 
        limit: int = 100
    ) -> List[ZetaMatch]:
        """Get best matching results."""
        return self.session.query(ZetaMatch).filter(
            ZetaMatch.experiment_id == experiment_id
        ).order_by(ZetaMatch.relative_error).limit(limit).all()


class ZetaZerosManager:
    """Manager for zeta function zeros cache."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def save_zeros(self, gamma_values: np.ndarray):
        """
        Save zeta zeros to cache.
        
        Args:
            gamma_values: Array of gamma_k values
        """
        existing = self.session.query(ZetaZerosCache.gamma_index).all()
        existing_indices = set(idx for idx, in existing)
        
        zeros_to_add = []
        for i, gamma in enumerate(gamma_values):
            if i not in existing_indices:
                zero = ZetaZerosCache(
                    gamma_index=i,
                    gamma_value=gamma,
                    frequency=gamma / (2 * np.pi)
                )
                zeros_to_add.append(zero)
        
        if zeros_to_add:
            self.session.bulk_save_objects(zeros_to_add)
            self.session.commit()
            print(f"Added {len(zeros_to_add)} new zeta zeros to cache")
    
    def get_all(self) -> np.ndarray:
        """Get all cached zeta zeros."""
        zeros = self.session.query(ZetaZerosCache).order_by(
            ZetaZerosCache.gamma_index
        ).all()
        return np.array([z.gamma_value for z in zeros])
    
    def get_frequencies(self) -> np.ndarray:
        """Get all cached frequencies (gamma_k / 2*pi)."""
        zeros = self.session.query(ZetaZerosCache).order_by(
            ZetaZerosCache.gamma_index
        ).all()
        return np.array([z.frequency for z in zeros])


class DatabaseManager:
    """
    Main database manager that combines all sub-managers.
    """
    
    def __init__(self, db_path: str, echo: bool = False):
        self.engine = create_database(db_path, echo=echo)
        self.session = get_session(self.engine)
        
        self.experiments = ExperimentManager(self.session)
        self.primes_metadata = PrimesMetadataManager(self.session)
        self.results = ResultManager(self.session)
        self.spectra = SpectrumManager(self.session)
        self.zeta_matches = ZetaMatchManager(self.session)
        self.zeta_zeros = ZetaZerosManager(self.session)
    
    def close(self):
        """Close database connection."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test database operations
    import os
    test_db = "test_operations.db"
    
    if os.path.exists(test_db):
        os.remove(test_db)
    
    with DatabaseManager(test_db) as db:
        # Create experiment
        exp = db.experiments.get_or_create(1000, 10.0, "Test")
        print(f"Created experiment: {exp.id}")
        
        # Save some test results
        primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
        need_values = np.random.rand(10) * 0.1
        cn_values = np.cumsum(need_values)
        
        db.results.save_batch(exp, primes, need_values=need_values, cn_values=cn_values)
        print(f"Saved {len(primes)} results")
        
        # Retrieve results
        retrieved_primes, retrieved_need, retrieved_cn, _ = db.results.get_arrays(exp.id)
        print(f"Retrieved {len(retrieved_primes)} primes")
    
    # Clean up
    os.remove(test_db)
    print("Test completed successfully!")
