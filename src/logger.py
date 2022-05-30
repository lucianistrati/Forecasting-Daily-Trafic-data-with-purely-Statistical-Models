import os
import logging

def log(path, file):

    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    handler = logging.FileHandler(log_file)

    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def log_results_regress(logger, model_name, logging_metrics_list):
    logger.info(model_name + " metrics results:")
    print(logging_metrics_list)
    for metric_name, metric_value in logging_metrics_list:
        logger.info("{}: {}".format(metric_name, metric_value))
    logger.info("-------------------------------")


def empty_regress_loggings():
    return [['R2', 0.0], ['MAPE', 0.0], ['MAE', 0.0], ['MSE', 0.0], ['MDA',
                                                                     0.0],
            ['MAD', 0.0]]

def main():
    pass

if __name__ == '__main__':
    main()