import os
import torch
import logging.config
import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column
import matplotlib as plt

###### For setting up for log.txt file##############


def setup_logging(log_file='log.txt'):
    """
    Setup logging configuration
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s-%(levelname)s-%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(messages)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
############### For csv file ################


class ResultsLog(object):
    def __init__(self, path='results.csv', plot_path=None):
        self.path = path
        self.plot_path = plot_path or (self.path + '.html')
        self.figures = []
        self.results = None

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            output_file(self.plot_path, title=title)
            plot = column(*self.figures)
            save(plot)
            self.figure = []
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.path
        if os.path.isfile(path):
            self.results.read_csv(path)

    def show(self):
        if len(self.figures > 0):
            plt.column(*self.figures)
            show(plt)

    def image(self, *kargs, **kwargs):
        fig = figure()
        fig.image(*kargs, **kwargs)
        self.figures.append(fig)


def init_path(cfg):
    results_save_path = cfg.get_save_path_F(
    ) if cfg.fg_PICNN_flag else cfg.get_save_path()
    model_save_path = results_save_path + '/storing_models'
    os.makedirs(model_save_path, exist_ok=True)
    setup_logging(os.path.join(results_save_path, 'log.txt'))
    results_file = os.path.join(results_save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')
    testresults_file = os.path.join(results_save_path, 'testresults.%s')
    testresults = ResultsLog(testresults_file %
                             'csv', testresults_file % 'html')
    logging.debug("run arguments:%s", cfg)
    return results_save_path, model_save_path, results, testresults


def dump_nn(generator_h, convex_f, convex_g, epoch, path, num_distribution=2, save_f=False):
    model_save_path = path
    torch.save(generator_h.state_dict(), model_save_path +
               '/generator_h_epoch{0}.pt'.format(epoch))
    for idx in range(num_distribution):
        torch.save(convex_g[idx].state_dict(), model_save_path +
                   f'/g{idx}_epoch{epoch}.pt')
        if save_f == True:
            torch.save(convex_f[idx].state_dict(), model_save_path +
                       f'/f{idx}_epoch{epoch}.pt')
