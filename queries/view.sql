CREATE VIEW kaggle_simple as (
    SELECT *
    FROM logging
    WHERE name in ('avg_train_loss', 'avg_valid_loss')
)