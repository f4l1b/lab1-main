"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors

    products = np.matmul(matrices, vectors)
    return np.sum(products, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    
    return [np.unique(col).tolist() for col in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns = data.rows, data.columns
    mean, std, seed = data.mean, data.std, data.seed
    
    rng = np.random.default_rng(seed)
    
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    
    row_means = np.mean(matrix, axis=1)
    col_means = np.mean(matrix, axis=0)
    row_vars = np.var(matrix, axis=1)
    col_vars = np.var(matrix, axis=0)

    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=col_means,
        row_variances=row_vars,
        column_variances=col_vars
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns = data.rows, data.columns
    first, second = data.first, data.second
    
    i = np.arange(rows).reshape(rows, 1)
    j = np.arange(columns)

    checkerboard = (i + j) % 2

    return np.where(checkerboard == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    rect_w, rect_h = data.width, data.height
    img_h, img_w = data.image_height, data.image_width
    shape_color, bg_color = data.shape_color, data.background_color
    
    # shape (img_h, img_w, 3)
    image = np.full((img_h, img_w, 3), bg_color, dtype=np.uint8)
    
    y_start = (img_h - rect_h) // 2
    y_end = y_start + rect_h
    x_start = (img_w - rect_w) // 2
    x_end = x_start + rect_w
    
    image[y_start:y_end, x_start:x_end] = shape_color
    
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    a, b = data.semi_axis_x, data.semi_axis_y
    img_h, img_w = data.image_height, data.image_width
    shape_color, bg_color = data.shape_color, data.background_color
    
    image = np.full((img_h, img_w, 3), bg_color, dtype=np.uint8)
    
    x0 = img_w / 2
    y0 = img_h / 2
    
    x = np.arange(img_w)
    y = np.arange(img_h)
    X, Y = np.meshgrid(x, y)
    
    ellipse_mask = ((X - x0) / a)**2 + ((Y - y0) / b)**2 <= 1
    
    image[ellipse_mask] = shape_color
    
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values = data.values
    window = data.window
    
    mean = np.mean(values)
    
    variance = np.var(values)
    
    std_dev = np.std(values)
    
    left_neighbors = values[:-2]
    current_elements = values[1:-1]
    right_neighbors = values[2:]
    
    is_max = (current_elements > left_neighbors) & (current_elements > right_neighbors)
    max_indices = np.where(is_max)[0] + 1
    
    is_min = (current_elements < left_neighbors) & (current_elements < right_neighbors)
    min_indices = np.where(is_min)[0] + 1
    
    window_array = np.ones(window) / window
    moving_average = np.convolve(values, window_array, mode='valid')
    
    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std_dev,
        max_indices=max_indices,
        min_indices=min_indices,
        moving_average=moving_average
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.asarray(data.labels)
    class_count = data.class_count
    
    if class_count is None:
        class_count = np.max(labels) + 1
        
    n_samples = len(labels)
    result = np.zeros((n_samples, class_count), dtype=int)
    
    result[np.arange(n_samples), labels] = 1
    
    return result
