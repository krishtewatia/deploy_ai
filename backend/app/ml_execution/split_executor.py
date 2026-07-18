"""Dataset split executor for DeployAI.

Stage 9B implements dataset partitioning using random, stratified, and
time-based strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.app.dataset_intelligence.schemas import DatasetContext
from backend.app.ml_plan.schemas import MLPlan, SplitStrategy, ProblemType


class SplitExecutorError(Exception):
    """Raised when dataset splitting fails due to invalid parameters or data issues."""

    pass


@dataclass
class DatasetSplitResult:
    """The structured result of partitioning a dataset."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: List[str]
    target_column: str
    train_indices: List[Any]
    test_indices: List[Any]


class SplitExecutor:
    """Performs dataset splitting according to the strategy defined in the MLPlan."""

    def execute(
        self,
        *,
        dataframe: pd.DataFrame,
        dataset_context: DatasetContext,
        plan: MLPlan,
    ) -> DatasetSplitResult:
        """Partition the input dataframe into train and test sets.

        Args:
            dataframe: Input pandas DataFrame to partition.
            dataset_context: Target dataset metadata profile.
            plan: The planned execution MLPlan.

        Returns:
            DatasetSplitResult containing partitioned X and y sets and indices.

        Raises:
            SplitExecutorError: For invalid inputs or operational partition failures.
        """
        # 1. Reject None inputs
        if dataframe is None:
            raise SplitExecutorError("dataframe cannot be None")
        if dataset_context is None:
            raise SplitExecutorError("dataset_context cannot be None")
        if plan is None:
            raise SplitExecutorError("plan cannot be None")

        # 2. Reject wrong types
        if not isinstance(dataframe, pd.DataFrame):
            raise SplitExecutorError("dataframe must be a pandas.DataFrame instance")
        if not isinstance(dataset_context, DatasetContext):
            raise SplitExecutorError("dataset_context must be a DatasetContext instance")
        if not isinstance(plan, MLPlan):
            raise SplitExecutorError("plan must be an MLPlan instance")

        # 3. Reject empty dataframe
        if dataframe.empty:
            raise SplitExecutorError("dataframe cannot be empty")

        # 4. Target column check
        target = plan.target_column
        if not target:
            raise SplitExecutorError("plan target_column cannot be empty")
        if target not in dataframe.columns:
            raise SplitExecutorError(f"Target column '{target}' does not exist in dataframe")

        # 5. Feature columns check
        features = plan.feature_columns
        if not features:
            raise SplitExecutorError("plan feature_columns cannot be empty")
        for col in features:
            if col not in dataframe.columns:
                raise SplitExecutorError(f"Feature column '{col}' does not exist in dataframe")

        # 6. Target inside feature columns check
        if target in features:
            raise SplitExecutorError("target_column cannot appear in feature_columns")

        # 7. Duplicate feature columns check
        if len(features) != len(set(features)):
            raise SplitExecutorError("plan feature_columns cannot contain duplicates")

        split_plan = plan.split_plan
        strategy = split_plan.strategy

        # 8. Strategy specific validations
        if strategy == SplitStrategy.STRATIFIED:
            stratify_col = split_plan.stratify_column
            if not stratify_col:
                raise SplitExecutorError("stratify_column is required for stratified split")
            if stratify_col not in dataframe.columns:
                raise SplitExecutorError(f"Stratify column '{stratify_col}' does not exist in dataframe")
            
            # Check problem definition inconsistency: stratified splits not allowed for regression
            if plan.problem_type == ProblemType.REGRESSION:
                raise SplitExecutorError("Stratified splits are not allowed for regression tasks")
            # stratify_column must match target_column for classification tasks
            if stratify_col != target:
                raise SplitExecutorError("stratify_column must match target_column for classification tasks")

        elif strategy == SplitStrategy.TIME_BASED:
            time_col = split_plan.time_column
            if not time_col:
                raise SplitExecutorError("time_column is required for time-based split")
            if time_col not in dataframe.columns:
                raise SplitExecutorError(f"Time column '{time_col}' does not exist in dataframe")

            # Check if time_column is datetime
            # Look up column type in DatasetContext or check in pandas DataFrame
            col_in_context = next((c for c in dataset_context.columns if c.name == time_col), None)
            if col_in_context is not None and not col_in_context.is_datetime:
                raise SplitExecutorError(f"Time column '{time_col}' must be a datetime column")

        # 9. Perform Split (safely operating on a copy of dataframe to prevent mutation)
        df_copy = dataframe.copy()
        X = df_copy[features]
        y = df_copy[target]

        try:
            if strategy == SplitStrategy.RANDOM:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=split_plan.test_size,
                    random_state=split_plan.random_state,
                    shuffle=split_plan.shuffle,
                )
            elif strategy == SplitStrategy.STRATIFIED:
                stratify_series = df_copy[split_plan.stratify_column]
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=split_plan.test_size,
                    random_state=split_plan.random_state,
                    shuffle=split_plan.shuffle,
                    stratify=stratify_series,
                )
            elif strategy == SplitStrategy.TIME_BASED:
                # Sort dataframe ascending by time column
                sorted_df = df_copy.sort_values(by=split_plan.time_column, ascending=True)
                n_samples = len(sorted_df)
                if n_samples < 2:
                    raise SplitExecutorError("Dataset too small to split train and test sets")
                
                # Test size fraction allocation
                test_samples = int(round(n_samples * split_plan.test_size))
                if test_samples < 1:
                    test_samples = 1
                if test_samples >= n_samples:
                    test_samples = n_samples - 1

                train_samples = n_samples - test_samples

                # Slice oldest for train, newest for test
                X_train = sorted_df.iloc[:train_samples][features]
                X_test = sorted_df.iloc[train_samples:][features]
                y_train = sorted_df.iloc[:train_samples][target]
                y_test = sorted_df.iloc[train_samples:][target]
            else:
                raise SplitExecutorError(f"Unsupported split strategy: {strategy}")

        except Exception as exc:
            if isinstance(exc, SplitExecutorError):
                raise exc
            raise SplitExecutorError(f"Split execution failed: {str(exc)}") from exc

        return DatasetSplitResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_columns=features,
            target_column=target,
            train_indices=list(X_train.index),
            test_indices=list(X_test.index),
        )
