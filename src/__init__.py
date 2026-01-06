"""
ToxicityShield - Deep Learning for Toxicity Detection in Game Chats
"""

from .data_processing import (
    TextPreprocessor,
    Vocabulary,
    pad_sequences,
    load_kaggle_data,
    prepare_data
)

from .models import (
    LSTMClassifier,
    GRUClassifier,
    TextCNN,
    MiniTransformer,
    get_model,
    count_parameters
)

from .training import (
    create_dataloaders,
    train_model,
    evaluate,
    get_full_metrics,
    print_metrics,
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_roc_curves_comparison,
    plot_model_comparison,
    ToxicityPredictor
)

__version__ = "1.0.0"
__author__ = "Markko"
