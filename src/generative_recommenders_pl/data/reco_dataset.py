import os
from typing import Dict, List, Optional, Tuple

import hydra
import lightning as L
import pandas as pd
import torch
from omegaconf import DictConfig

from generative_recommenders_pl.data.preprocessor import DataProcessor
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)


def load_data(ratings_file: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(ratings_file, pd.DataFrame):
        return ratings_file
    elif isinstance(ratings_file, str) and ratings_file.endswith(".csv"):
        return pd.read_csv(ratings_file, delimiter=",")
    else:
        raise ValueError("ratings_file must be a csv file.")


def save_data(ratings_frame: pd.DataFrame, output_file: str):
    if output_file.endswith(".csv"):
        ratings_frame.to_csv(output_file, index=False)
    else:
        raise ValueError("ratings_file must be a csv file.")


class RecoDataset(torch.utils.data.Dataset):
    """In reverse chronological order."""

    def __init__(
        self,
        ratings_file: str | pd.DataFrame,
        padding_length: int,
        ignore_last_n: int,  # used for creating train/valid/test sets
        shift_id_by: int = 0,
        chronological: bool = False,
        sample_ratio: float = 1.0,
        additional_columns: Optional[List[str]] = [],
    ) -> None:
        """
        Args:
            ratings_file: str or pd.DataFrame, path to the ratings file or DataFrame.
            padding_length: int, length to pad sequences to.
            ignore_last_n: int, number of last interactions to ignore (used for creating train/valid/test sets).
            shift_id_by: int, value to shift IDs by. Default is 0.
            chronological: bool, whether to sort interactions chronologically. Default is False.
            sample_ratio: float, ratio of data to sample. Default is 1.0 (use all data).
            additional_columns: Optional[List[str]], list of additional columns to include. Default is None.
        """
        super().__init__()

        self.ratings_frame: pd.DataFrame = load_data(ratings_file)
        self._padding_length: int = padding_length
        self._ignore_last_n: int = ignore_last_n
        self._cache = dict()
        self._shift_id_by: int = shift_id_by
        self._chronological: bool = chronological
        self._sample_ratio: float = sample_ratio
        self._additional_columns = additional_columns
        self.__additional_columns_check()

    def __additional_columns_check(self):
        if self._additional_columns:
            columns_status = []
            for column in self._additional_columns:
                # check the column exists and status, like type, max, min, etc.
                column_exists = column in self.ratings_frame.columns
                if not column_exists:
                    raise ValueError(
                        f"Column {column} does not exist in the ratings file."
                    )
                column_type = self.ratings_frame[column].dtype
                max_value = self.ratings_frame[column].max()
                min_value = self.ratings_frame[column].min()
                columns_status.append(
                    {
                        "column": column,
                        "type": column_type,
                        "max": max_value,
                        "min": min_value,
                    }
                )
            log.info(f"Additional columns status: {columns_status}")

    def __len__(self) -> int:
        return len(self.ratings_frame)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if idx in self._cache.keys():
            return self._cache[idx]
        sample = self.load_item(idx)
        self._cache[idx] = sample
        return sample

    def load_item(self, idx) -> Dict[str, torch.Tensor]:
        data = self.ratings_frame.iloc[idx]
        user_id = data.user_id

        def eval_as_list(x, ignore_last_n) -> List[int]:
            y = eval(x)
            y_list = [y] if isinstance(y, int) else list(y)
            if ignore_last_n > 0:
                # for training data creation
                y_list = y_list[:-ignore_last_n]
            return y_list

        def eval_int_list(
            x,
            target_len: int,
            ignore_last_n: int,
            shift_id_by: int,
            sampling_kept_mask: Optional[List[bool]],
        ) -> Tuple[List[int], int]:
            y = eval_as_list(x, ignore_last_n=ignore_last_n)
            if sampling_kept_mask is not None:
                y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
            y_len = len(y)
            y.reverse()
            if shift_id_by > 0:
                y = [x + shift_id_by for x in y]
            return y, y_len

        if self._sample_ratio < 1.0:
            raw_length = len(eval_as_list(data.sequence_item_ids, self._ignore_last_n))
            sampling_kept_mask = (
                torch.rand((raw_length,), dtype=torch.float32) < self._sample_ratio
            ).tolist()
        else:
            sampling_kept_mask = None

        movie_history, movie_history_len = eval_int_list(
            data.sequence_item_ids,
            self._padding_length,
            self._ignore_last_n,
            shift_id_by=self._shift_id_by,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_history_ratings, ratings_len = eval_int_list(
            data.sequence_ratings,
            self._padding_length,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_timestamps, timestamps_len = eval_int_list(
            data.sequence_timestamps,
            self._padding_length,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        assert (
            movie_history_len == timestamps_len
        ), f"history len {movie_history_len} differs from timestamp len {timestamps_len}."
        assert (
            movie_history_len == ratings_len
        ), f"history len {movie_history_len} differs from ratings len {ratings_len}."

        def _truncate_or_pad_seq(
            y: List[int], target_len: int, chronological: bool
        ) -> List[int]:
            y_len = len(y)
            if y_len < target_len:
                y = y + [0] * (target_len - y_len)
            else:
                if not chronological:
                    y = y[:target_len]
                else:
                    y = y[-target_len:]
            assert len(y) == target_len
            return y

        historical_ids = movie_history[1:]
        historical_ratings = movie_history_ratings[1:]
        historical_timestamps = movie_timestamps[1:]
        target_ids = movie_history[0]
        target_ratings = movie_history_ratings[0]
        target_timestamps = movie_timestamps[0]
        if self._chronological:
            historical_ids.reverse()
            historical_ratings.reverse()
            historical_timestamps.reverse()

        max_seq_len = self._padding_length - 1
        history_length = min(len(historical_ids), max_seq_len)
        historical_ids = _truncate_or_pad_seq(
            historical_ids,
            max_seq_len,
            self._chronological,
        )
        historical_ratings = _truncate_or_pad_seq(
            historical_ratings,
            max_seq_len,
            self._chronological,
        )
        historical_timestamps = _truncate_or_pad_seq(
            historical_timestamps,
            max_seq_len,
            self._chronological,
        )
        ret = {
            "user_id": user_id,
            "historical_ids": torch.tensor(historical_ids, dtype=torch.int64),
            "historical_ratings": torch.tensor(historical_ratings, dtype=torch.int64),
            "historical_timestamps": torch.tensor(
                historical_timestamps, dtype=torch.int64
            ),
            "history_lengths": history_length,
            "target_ids": target_ids,
            "target_ratings": target_ratings,
            "target_timestamps": target_timestamps,
        }

        for column in self._additional_columns:
            # currently we do not consider the sequence columns in the additional columns
            ret[column] = data[column]
        return ret


class RecoDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        data_preprocessor: DataProcessor,
        train_dataset: RecoDataset | DictConfig,
        val_dataset: RecoDataset | DictConfig,
        test_dataset: RecoDataset | DictConfig,
        max_sequence_length: int,
        chronological: bool,
        positional_sampling_ratio: float,
        batch_size: int = 32,
        num_workers: int = os.cpu_count() // 4,
        prefetch_factor: int = 4,
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.dataset_name = dataset_name
        self.data_preprocessor: DataProcessor = (
            hydra.utils.instantiate(data_preprocessor)
            if isinstance(data_preprocessor, DictConfig)
            else data_preprocessor
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.max_sequence_length = max_sequence_length
        self.chronological = chronological
        self.positional_sampling_ratio = positional_sampling_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch = prefetch_factor
        self.__init_item_ids()

    def __init_item_ids(self):
        if self.dataset_name == "ml-1m" or self.dataset_name == "ml-20m":
            items = pd.read_csv(
                self.data_preprocessor.processed_item_csv(), delimiter=","
            )
            max_jagged_dimension = 16
            max_item_id = self.data_preprocessor.expected_max_item_id()

            # Initialize dictionaries for lengths and values
            lengths = {
                i: torch.zeros((max_item_id + 1,), dtype=torch.int64) for i in range(3)
            }
            values = {
                i: torch.zeros(
                    (max_item_id + 1, max_jagged_dimension), dtype=torch.int64
                )
                for i in range(3)
            }

            # Define max index ranges for each feature type
            max_ind_ranges = [63, 16383, 511]

            all_item_ids = []
            for df_index, row in items.iterrows():
                movie_id = int(row["movie_id"])
                genres = row["genres"].split("|")
                titles = row["cleaned_title"].split(" ")
                years = [row["year"]]

                # Process each feature type
                for i, feature_set in enumerate([genres, titles, years]):
                    feature_vector = [hash(x) % max_ind_ranges[i] for x in feature_set]
                    lengths[i][movie_id] = min(
                        len(feature_vector), max_jagged_dimension
                    )
                    for j, value in enumerate(feature_vector[:max_jagged_dimension]):
                        values[i][movie_id][j] = value

                all_item_ids.append(movie_id)
            self.all_item_ids = all_item_ids
            self.max_item_id = max_item_id
        else:
            self.all_item_ids = [
                x + 1 for x in range(self.data_preprocessor.expected_num_unique_items())
            ]
            self.max_item_id = self.data_preprocessor.expected_num_unique_items()

    def instantiate_dataset(self, dataset: RecoDataset | DictConfig) -> RecoDataset:
        if isinstance(dataset, DictConfig):
            kwargs = {}
            if "padding_length" not in dataset:
                kwargs["padding_length"] = self.max_sequence_length + 1
            if "chronological" not in dataset:
                kwargs["chronological"] = self.chronological
            if "position_sampling_ratio" not in dataset:
                kwargs["sample_ratio"] = self.positional_sampling_ratio
            # preload the data for shared dataset
            ratings_file = (
                dataset.pop("ratings_file")
                if "ratings_file" in dataset
                else self.data_preprocessor.output_format_csv()
            )
            ratings_file = load_data(ratings_file)
            return hydra.utils.instantiate(dataset, ratings_file=ratings_file, **kwargs)
        else:
            return dataset

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.instantiate_dataset(self.train_dataset)
            self.val_dataset = self.instantiate_dataset(self.val_dataset)

        if stage == "test" or stage == "predict" or stage is None:
            self.test_dataset = self.instantiate_dataset(self.test_dataset)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch,
        )

    def save_predictions(self, output_file: str, predictions: dict):
        """Save the predictions to a file.

        It adds the predictions to the ratings_frame in the test dataset
        since it is used for prediction and saves it to a file. And it
        expects the predictions to be a dictionary of list / numpy arrays,
        which has the same length and order as the test dataset.

        Args:
            output_file: str, path to the output file.
            predictions: dict, predictions to save.
        """
        ratings_frame = self.test_dataset.ratings_frame
        for key, value in predictions.items():
            ratings_frame[key] = value
        save_data(ratings_frame, output_file)
