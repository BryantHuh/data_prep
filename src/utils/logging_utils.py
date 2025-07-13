#!/usr/bin/env python3
"""
Logging utilities for BCI system

This module provides standardized logging setup and utilities for all BCI scripts.
"""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(name: str, log_dir: str = 'logs', level: str = 'INFO') -> logging.Logger:
    """
    Setup logging for a component with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # Create file handler
    log_file = Path(log_dir) / f"{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    return logger


def log_training_metrics(logger: logging.Logger, epoch: int, train_loss: float,
                        val_loss: float, train_acc: float, val_acc: float):
    """
    Log training metrics in a standardized format.

    Args:
        logger: Logger instance
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        train_acc: Training accuracy
        val_acc: Validation accuracy
    """
    logger.info(
        f"Epoch {epoch:3d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}%"
    )


def log_evaluation_metrics(logger: logging.Logger, accuracy: float, precision: float,
                          recall: float, f1_score: float):
    """
    Log evaluation metrics in a standardized format.

    Args:
        logger: Logger instance
        accuracy: Overall accuracy
        precision: Weighted precision
        recall: Weighted recall
        f1_score: Weighted F1 score
    """
    logger.info(
        f"Evaluation Results | "
        f"Accuracy: {accuracy:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1-Score: {f1_score:.4f}"
    )


def log_performance_stats(logger: logging.Logger, avg_inference_time: float,
                         total_samples: int, total_time: float):
    """
    Log performance statistics.

    Args:
        logger: Logger instance
        avg_inference_time: Average inference time per sample
        total_samples: Total number of samples processed
        total_time: Total processing time
    """
    logger.info(
        f"Performance Stats | "
        f"Avg Inference: {avg_inference_time:.4f}s | "
        f"Total Samples: {total_samples} | "
        f"Total Time: {total_time:.2f}s"
    )


def create_logger_with_config(name: str, config: dict) -> logging.Logger:
    """
    Create logger with configuration from a config dictionary.

    Args:
        name: Logger name
        config: Configuration dictionary with log_dir and log_level keys

    Returns:
        Configured logger instance
    """
    log_dir = config.get('log_dir', 'logs')
    log_level = config.get('log_level', 'INFO')

    return setup_logging(name, log_dir, log_level)