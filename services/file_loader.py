"""Validated, framework-independent CSV and Excel loading."""

from __future__ import annotations

import hashlib
import io
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class FileLoadError(ValueError):
    """A safe, user-facing dataset loading error."""


@dataclass(frozen=True)
class LoadedDataset:
    """A loaded dataset and its source metadata."""

    dataframe: pd.DataFrame
    filename: str
    file_size: int
    fingerprint: str
    sheet_name: str | None = None


def read_file_bytes(file_obj: BinaryIO | bytes) -> bytes:
    """Read uploaded bytes while preserving seekable file position."""
    if isinstance(file_obj, bytes):
        return file_obj
    position = file_obj.tell() if hasattr(file_obj, "tell") else None
    data = file_obj.read()
    if position is not None and hasattr(file_obj, "seek"):
        file_obj.seek(position)
    return data


def validate_upload(filename: str, data: bytes, max_size_mb: int) -> str:
    """Validate filename, type, size, and non-empty content."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise FileLoadError("Unsupported file type. Upload a CSV, XLSX, or XLS file.")
    if not data:
        raise FileLoadError("The uploaded file is empty.")
    if len(data) > max_size_mb * 1024 * 1024:
        raise FileLoadError(f"The file exceeds the {max_size_mb} MB upload limit.")
    return extension


def fingerprint_file(data: bytes) -> str:
    """Create a stable dataset identifier without retaining file contents."""
    return hashlib.sha256(data).hexdigest()


def get_excel_sheet_names(filename: str, data: bytes, max_size_mb: int = 100) -> list[str]:
    """Return workbook sheet names after validating the upload."""
    validate_upload(filename, data, max_size_mb)
    try:
        with pd.ExcelFile(io.BytesIO(data)) as workbook:
            sheets = workbook.sheet_names
    except ImportError as exc:
        raise FileLoadError("This Excel format requires an additional reader dependency.") from exc
    except Exception as exc:
        raise FileLoadError("The Excel workbook could not be read. Check that it is valid.") from exc
    if not sheets:
        raise FileLoadError("The Excel workbook contains no worksheets.")
    return sheets


def _read_csv(data: bytes) -> pd.DataFrame:
    errors: list[Exception] = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(exc)
        except pd.errors.EmptyDataError as exc:
            raise FileLoadError("The CSV contains no readable columns.") from exc
        except pd.errors.ParserError as exc:
            raise FileLoadError("The CSV is malformed and could not be parsed.") from exc
    raise FileLoadError("The CSV encoding could not be detected.") from errors[-1]


def _validate_csv_headers(data: bytes) -> None:
    """Inspect the raw CSV header before pandas normalizes duplicate names."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            header = next(csv.reader(io.StringIO(data.decode(encoding))), [])
            break
        except UnicodeDecodeError:
            continue
    else:
        return
    normalized = [column.strip() for column in header]
    if not normalized or any(not column for column in normalized):
        raise FileLoadError("The CSV contains missing or invalid column headers.")
    if len(set(normalized)) != len(normalized):
        raise FileLoadError("Duplicate column names were detected. Rename them before uploading.")


def _validate_dataframe(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        raise FileLoadError("The dataset contains no data rows.")
    if len(dataframe.columns) == 0:
        raise FileLoadError("The dataset contains no usable columns.")
    unnamed = [str(column) for column in dataframe.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        raise FileLoadError("The dataset does not appear to contain valid column headers.")


def load_dataset(
    filename: str,
    data: bytes,
    *,
    sheet_name: str | None = None,
    max_size_mb: int = 100,
) -> LoadedDataset:
    """Load a validated CSV or selected Excel worksheet."""
    extension = validate_upload(filename, data, max_size_mb)
    try:
        if extension == ".csv":
            _validate_csv_headers(data)
            dataframe = _read_csv(data)
            active_sheet = None
        else:
            sheets = get_excel_sheet_names(filename, data, max_size_mb)
            active_sheet = sheet_name or sheets[0]
            if active_sheet not in sheets:
                raise FileLoadError(f'Worksheet "{active_sheet}" was not found in the workbook.')
            dataframe = pd.read_excel(io.BytesIO(data), sheet_name=active_sheet)
    except FileLoadError:
        raise
    except Exception as exc:
        raise FileLoadError("The dataset could not be read. Verify the file contents.") from exc

    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    if len(set(dataframe.columns)) != len(dataframe.columns):
        raise FileLoadError("Duplicate column names were detected. Rename them before uploading.")
    _validate_dataframe(dataframe)
    return LoadedDataset(
        dataframe=dataframe,
        filename=filename,
        file_size=len(data),
        fingerprint=fingerprint_file(data),
        sheet_name=active_sheet,
    )
