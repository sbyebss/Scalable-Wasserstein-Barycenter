from dataclasses import dataclass, field
import pickle
from pathlib import Path


@dataclass
class Cfg2layer:
    # N_TRAIN_SAMPLES: int = 2600
    NUM_DISTRIBUTION: int = 5
    # optimizer: str = 'Adam'
    BATCH_SIZE: int = 100
    LR: float = 1e-3
    INPUT_DIM: int = 785
    OUTPUT_DIM: int = 1
    log_interval: int = 10
    N_TEST: int = 10000
    activation: str = 'celu'
    final_actv: str = 'sigmoid'
    schedule_learning_rate: bool = True
    lr_schedule_scale: float = 0.1
    lr_schedule_per_epoch: int = 20
    epochs: int = 200
    NUM_NEURON: int = 1024
    NUM_LAYERS: int = 0
    TRIAL: int = 26
    two_digit: int = 17
    idx_subset: int = 1
    train_data_path: str = 'input_data/mnist_data'
    test_data_path: str = 'input_data/mnist_data/mnist_17_classi_test_train.pt'
    repeat: int = 0
    subset_samples: int = 500

    def get_save_path(self):
        return './data/Results_of_classification/distribution_{7}/digit{11}/input_dim_{5}/layers_{0}/neuron_{1}/activ_{9}_final_{6}/lr{2}/schedule_learning_rate:{8}/lr_schedule:{10}/batch_{3}/subset_{12}_samples_{13}/trial_{4}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON,
            self.LR,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            self.final_actv,
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            self.activation,
            self.lr_schedule_per_epoch,
            self.two_digit,
            self.idx_subset,
            13000 if self.idx_subset == 0 else self.subset_samples
        )


@dataclass
class CfgBase:
    NUM_DISTRIBUTION: int = 1
    optimizer: str = 'Adam'
    BATCH_SIZE: int = 100
    LR_g: float = 1e-3
    LR_f: float = 1e-3
    INPUT_DIM: int = 2
    INPUT_DIM_fg: int = 2
    LAMBDA_CVX: float = 0.1
    log_interval: int = 10
    N_TEST: int = 10000
    fg_activation: str = 'celu'
    schedule_learning_rate: bool = True
    lr_schedule_scale: float = 0.1
    lr_schedule_per_epoch: int = 20
    high_dim_flag: bool = False
    epochs: int = 80
    NUM_NEURON: int = 32
    NUM_LAYERS: int = 3
    TRIAL: int = 4
    N_Gnet_ITERS: int = 6
    load_h: int = 0
    load_fg: int = 0
    load_epoch: int = 80
    repeat: int = 0


@dataclass
class Cfg3loop(CfgBase):
    fg_PICNN_flag: int = 0
    TRIAL: float = 20.2
    N_TRAIN_SAMPLES: int = 60000
    NUM_LAYERS_h: int = 3
    NUM_NEURON_h: int = 32
    OUTPUT_DIM: int = 2
    batch_nml: int = 1
    dropout: bool = False
    h_full_activation: bool = True
    N_Fnet_ITERS: int = 4
    schedule_hchange_bool: int = 0
    schedule_hchange_epoch: int = 0
    hLinear: bool = False
    hResNet: bool = False
    h_activation: str = 'Prelu'
    final_actv: str = 'Prelu'
    LR_h: float = 1e-3
    save_f: int = 0
    opacity: float = 1
    scatter_size: float = 10
    type_data: str = 'usps_mnist'

    seed: int = 1
    tag: str = "Empty"

    def save_point(self, i):
        _path = Path(f"error_bar_exp/{self.tag}")
        if not _path.exists():
            _path.mkdir(parents=True, exist_ok=True)
        with open(f"{_path}/{self.seed}.pkl", 'wb') as handle:
            pickle.dump(i, handle)

    def load_point(self):
        _path = Path(f"error_bar_exp/{self.tag}")
        with open(f"{_path}/{self.seed}.pkl", 'rb') as handle:
            i = pickle.load(handle)
        return i


@dataclass
class Cfg3loop_F(Cfg3loop):
    choice_weight: int = 1
    # '0 represents average weight, 1 represents random uniform weights, 2 represents fixed uniform weights'

    h_PICNN_flag: int = 0
    # 'if PICNN has point-wise product or not'
    expanded_PICNN: int = 0
    # 'number of neurons per layer for sample input path in f ang g'
    NUM_NEURON_fg_sample: int = 6
    # 'number of neurons per hidden layer for weight path in f ang g'
    NUM_NEURON_fg_weight: int = 6
    # 'number of neurons per layer for sample input path in h'
    NUM_NEURON_h_sample: int = 12
    # 'number of neurons per hidden layer for weight path in h'
    NUM_NEURON_h_weight: int = -1


@dataclass
class CfgNN(Cfg3loop):
    N_TRAIN_SAMPLES: int = 1024
    INPUT_DIM: int = 786
    INPUT_DIM_fg: int = 786
    OUTPUT_DIM: int = 786
    NUM_NEURON: int = 1024
    NUM_NEURON_h: int = 1024
    TRIAL: float = 27
    NUM_DISTRIBUTION: int = 5
    N_TEST: int = 10
    SCALE: int = 1000
    test_data_path: str = 'input_data/mnist_data/mnist_17_classi_test_train.pt'

    def get_nn(self):
        return f'data/Results_of_classification/distribution_5/digit17/input_dim_785/layers_0/neuron_{self.N_TRAIN_SAMPLES}/activ_celu_final_sigmoid/lr0.001/schedule_learning_rate:Yes/lr_schedule:20/batch_100'

    def get_save_path(self):
        return './data/Results_of_NN/distribution_{17}/input_dim_{5}/fg_FICNN/h_{23}/h_batchnml:{8}_dropout:{16}/layers_fg_{0}_h_{22}/neuron_fg{1}_h{21}/h_activ_{11}/lr_g{2}lr_f_{14}lr_h{15}/schedule_learning_rate:{18}/lr_schedule:{25}/gIterate_{10}_fIterate_{20}/batch_{3}/train_sample_{7}/scale_{12}/trial_{4}_last_{13}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON,
            self.LR_g,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            0,
            self.N_TRAIN_SAMPLES,
            'Yes' if self.batch_nml else 'No',
            self.optimizer,
            self.N_Gnet_ITERS,
            self.h_activation,
            self.SCALE,
            self.final_actv,
            self.LR_f,
            self.LR_h,
            'Yes' if self.dropout else 'No',
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            'Yes' if self.hLinear else 'No',
            self.N_Fnet_ITERS,
            self.NUM_NEURON_h,
            self.NUM_LAYERS_h,
            'ResNet' if self.hResNet else 'nonResNet',
            0,
            self.lr_schedule_per_epoch
        )


@dataclass
class CfgBayesian(Cfg3loop):
    N_TRAIN_SAMPLES: int = 10000
    TRIAL = 25
    posterior_path: str = "/home/jfan97/Study_hard/barycenter/July20/barycenter_clean/input_data/bayesian_inference/bike_posterior/"
    INPUT_DIM: int = 8
    INPUT_DIM_fg: int = 8
    OUTPUT_DIM: int = 8
    NUM_DISTRIBUTION: int = 5
    SCALE: float = 100
    BATCH_SIZE: int = 100

    def get_save_path(self):
        return './data/Results_of_bayesian/h_Linear:{19}_{23}/distribution_{17}/input_dim_{5}/fg_FICNN/h_batchnml:{8}_dropout:{16}/layers_fg_{0}_h_{22}/neuron_{1}/h_activ_{11}_fg_actic_{6}/lr_g{2}lr_f_{14}lr_h{15}/schedule_learning_rate:{18}/lr_schedule:{25}/gIterate_{10}_fIterate_{20}/batch_{3}/trial_{4}/scale_{7}/_last_{13}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON,
            self.LR_g,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            self.fg_activation,
            round(self.SCALE),
            'Yes' if self.batch_nml else 'No',
            self.optimizer,
            self.N_Gnet_ITERS,
            self.h_activation,
            0,
            'hFull_activation' if self.h_full_activation else 'hFinal_linear',
            self.LR_f,
            self.LR_h,
            'Yes' if self.dropout else 'No',
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            'Yes' if self.hLinear else 'No',
            self.N_Fnet_ITERS,
            0,
            self.NUM_LAYERS_h,
            'ResNet' if self.hResNet else 'Normal',
            0,
            self.lr_schedule_per_epoch
        )


@dataclass
class CfgGMM(Cfg3loop_F):
    NUM_GMM_COMPONENT: list = field(default_factory=list)
    MEAN: list = field(default_factory=list)
    COV: list = field(default_factory=list)
    NUM_NEURON: int = 32
    NUM_NEURON_h: int = 32
    INPUT_DIM_fg: int = 0
    NUM_LAYERS: int = 3
    NUM_LAYERS_h: int = 3
    epochs: int = 40

    def get_save_path(self):
        return './data/Results_of_Gauss2Gauss/{24}/h_Linear:{19}_{23}/distribution_{17}/GMM_component_{21}/input_dim_{5}/fg_FICNN/h_batchnml:{8}_dropout:{16}/layers_fg_{0}_h_{22}/neuron_fg{1}_h{6}/h_activ_{11}/lr_g{2}lr_f_{14}lr_h{15}/schedule_learning_rate:{18}/lr_schedule:{25}/gIterate_{10}_fIterate_{20}/batch_{3}/trial_{4}_last_{13}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON,
            self.LR_g,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            self.NUM_NEURON_h,
            0,
            'Yes' if self.batch_nml else 'No',
            self.optimizer,
            self.N_Gnet_ITERS,
            self.h_activation,
            0,
            'hFull_activation' if self.h_full_activation else 'hFinal_linear',
            self.LR_f,
            self.LR_h,
            'Yes' if self.dropout else 'No',
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            'Yes' if self.hLinear else 'No',
            self.N_Fnet_ITERS,
            self.NUM_GMM_COMPONENT,
            self.NUM_LAYERS_h,
            'ResNet' if self.hResNet else 'Normal',
            'high_dimension' if self.high_dim_flag else 'low_dimension',
            self.lr_schedule_per_epoch
        )

    def get_save_path_F(self):
        return './data/Results_of_Gauss2Gauss/{24}/h_Linear:{19}_{23}/distribution_{17}/GMM_component_{21}/input_dim_{5}/fg_{7}_{9}/h_batchnml:{16}_dropout{27}/layers_fg_{0}_h_{22}/neuron_fg_s{1}_w{8}_h_s{12}_w{25}/h_activ_{11}/lr_g{2}lr_f_{14}lr_h{15}/schedule_learning_rate:{18}/lr_schedule:{26}/gIterate_{10}_fIterate_{20}/batch_{3}/trial_{4}_last_{13}/epoch{6}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON_fg_sample,
            self.LR_g,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            self.epochs,
            'PICNN',
            self.NUM_NEURON_fg_weight,
            'expand' if self.expanded_PICNN else 'not-expand',
            self.N_Gnet_ITERS,
            self.h_activation,
            self.NUM_NEURON_h_sample if self.h_PICNN_flag else self.NUM_NEURON_h,
            'hFull_activation' if self.h_full_activation else 'hFinal_linear',
            self.LR_f,
            self.LR_h,
            'Yes' if self.batch_nml else 'No',
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            'Yes' if self.hLinear else 'No',
            self.N_Fnet_ITERS,
            self.NUM_GMM_COMPONENT,
            self.NUM_LAYERS_h,
            'ResNet' if self.hResNet else 'Normal',
            'high_dimension' if self.high_dim_flag else 'low_dimension',
            self.NUM_NEURON_h_weight if self.h_PICNN_flag else 'None',
            self.lr_schedule_per_epoch,
            'Yes' if self.dropout else 'No'
        )


@dataclass
class CfgMnist(Cfg3loop_F):
    epochs: int = 500
    BATCH_SIZE: int = 100
    N_TEST: int = 900
    INPUT_DIM: int = 32
    INPUT_DIM_fg: int = 784
    dropout: int = 1
    lr_schedule_per_epoch: int = 100
    OUTPUT_DIM: int = 784
    high_dim_flag: bool = True
    NUM_NEURON: int = 1024
    NUM_NEURON_h: int = 1024
    NUM_LAYERS_h: int = 2
    TRIAL: float = 20.1
    NUM_DISTRIBUTION: int = 2
    convolution_flag: bool = False
    final_actv: str = "tanh"
    LR_h: float = 1e-3
    LR_f: float = 1e-4
    LR_g: float = 1e-4
    N_TRAIN_SAMPLES: int = 1500
    positive_flag: int = 0
    num_digit: int = 0
    mnist_data_path: str = "./input_data/mnist_data"
    two_digit: int = 10

    def get_save_path(self):
        return './data/Results_of_MNIST/distribution_{17}/{24}/h_input_dim_{5}/fg_FICNN/h_{23}/h_batchnml:{8}_dropout:{16}/convolution_{6}/layers_fg_{0}_h_{22}/neuron_fg{1}_h{21}/h_activ_{11}/lr_g{2}lr_f_{14}lr_h{15}/schedule_learning_rate:{18}/lr_schedule:{25}/gIterate_{10}_fIterate_{20}/batch_{3}/train_sample_{7}/sign_{12}/trial_{4}_last_{13}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON,
            self.LR_g,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            'Yes' if self.convolution_flag else 'No',
            self.N_TRAIN_SAMPLES,
            'Yes' if self.batch_nml else 'No',
            self.optimizer,
            self.N_Gnet_ITERS,
            self.h_activation,
            self.positive_flag,
            self.final_actv if self.h_full_activation else 'linear',
            self.LR_f,
            self.LR_h,
            'Yes' if self.dropout else 'No',
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            'Yes' if self.hLinear else 'No',
            self.N_Fnet_ITERS,
            self.NUM_NEURON_h,
            self.NUM_LAYERS_h,
            'ResNet' if self.hResNet else 'nonResNet',
            f'{self.two_digit}bryc' if self.NUM_DISTRIBUTION > 1 else 'GAN',
            self.lr_schedule_per_epoch
        )

    def get_save_path2(self):
        return './data/Results_of_MNIST/distribution_{17}/{24}/h_input_dim_{5}/fg_FICNN/h_{23}/h_batchnml:{8}_dropout:{16}/convolution_{6}/layers_fg_{0}_h_{22}/neuron_fg{1}_h{21}/h_activ_{11}/lr_g{2}lr_f_{14}lr_h{15}/schedule_learning_rate:{18}/lr_schedule:{25}/gIterate_{10}_fIterate_{20}/batch_{3}/train_sample_{7}/sign_{12}/trial_{4}_last_{13}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON,
            self.LR_g,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            'Yes' if self.convolution_flag else 'No',
            self.N_TRAIN_SAMPLES,
            'Yes' if self.batch_nml else 'No',
            self.optimizer,
            self.N_Gnet_ITERS,
            self.h_activation,
            self.positive_flag,
            'hFull_activation' if self.h_full_activation else 'linear',
            self.LR_f,
            self.LR_h,
            'Yes' if self.dropout else 'No',
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            'Yes' if self.hLinear else 'No',
            self.N_Fnet_ITERS,
            self.NUM_NEURON_h,
            self.NUM_LAYERS_h,
            'ResNet' if self.hResNet else 'nonResNet',
            f'{self.two_digit}bryc' if self.NUM_DISTRIBUTION > 1 else 'GAN',
            self.lr_schedule_per_epoch
        )


@dataclass
class Cfg3digit(CfgMnist):
    opacity: float = 0.5
    scatter_size: float = 1
    epochs: int = 80
    INPUT_DIM: int = 2
    INPUT_DIM_fg: int = 2
    OUTPUT_DIM: int = 2
    N_TRAIN_SAMPLES: int = 60000
    lr_schedule_per_epoch: int = 20
    num_digit: int = 1
    TRIAL: float = 17.4
    NUM_NEURON: int = 10
    NUM_NEURON_h: int = 10
    high_dim_flag: bool = False

    def get_save_path(self):
        return './data/Results_of_MNIST/distribution_{17}/3digit/h_input_dim_{5}/fg_FICNN/h_{23}/h_batchnml:{8}_dropout:{16}/convolution_{6}/layers_fg_{0}_h_{22}/neuron_fg{1}_h{21}/h_activ_{11}/lr_g{2}lr_f_{14}lr_h{15}/schedule_learning_rate:{18}/lr_schedule:{25}/gIterate_{10}_fIterate_{20}/batch_{3}/train_sample_{7}/sign_{12}/trial_{4}_last_{13}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON,
            self.LR_g,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            'Yes' if self.convolution_flag else 'No',
            self.N_TRAIN_SAMPLES,
            'Yes' if self.batch_nml else 'No',
            self.optimizer,
            self.N_Gnet_ITERS,
            self.h_activation,
            self.positive_flag,
            'hFull_activation' if self.h_full_activation else 'hFinal_linear',
            self.LR_f,
            self.LR_h,
            'Yes' if self.dropout else 'No',
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            'Yes' if self.hLinear else 'No',
            self.N_Fnet_ITERS,
            self.NUM_NEURON_h,
            self.NUM_LAYERS_h,
            'ResNet' if self.hResNet else 'nonResNet',
            0,
            self.lr_schedule_per_epoch
        )


@dataclass
class CfgGAN(CfgMnist):
    epochs: int = 500
    BATCH_SIZE: int = 100
    INPUT_DIM: int = 32
    NUM_LAYERS_h: int = 4
    NUM_DISTRIBUTION: int = 1
    lr_schedule_per_epoch: int = 100
    label_result: str = "??"

    def get_save_path(self):
        return './data/Results_of_GAN/{24}/distribution_{17}/h_input_dim_{5}/fg_FICNN/h_{23}/h_batchnml:{8}_dropout:{16}/convolution_{6}/layers_fg_{0}_h_{22}/neuron_fg{1}_h{21}/h_activ_{11}/lr_g{2}_lr_f{14}_lr_h{15}/schedule_learning_rate:{18}/lr_schedule:{25}/gIterate_{10}_fIterate_{20}/batch_{3}/train_sample_{7}/trial_{4}_last_{13}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON,
            self.LR_g,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            'Yes' if self.convolution_flag else 'No',
            self.N_TRAIN_SAMPLES,
            'Yes' if self.batch_nml else 'No',
            0,
            self.N_Gnet_ITERS,
            self.h_activation,
            0,
            self.final_actv if self.h_full_activation else 'linear',
            self.LR_f,
            self.LR_h,
            'Yes' if self.dropout else 'No',
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            'Yes' if self.hLinear else 'No',
            self.N_Fnet_ITERS,
            self.NUM_NEURON_h,
            self.NUM_LAYERS_h,
            'ResNet' if self.hResNet else 'nonResNet',
            self.label_result,
            self.lr_schedule_per_epoch
        )


@dataclass
class CfgCifar(CfgGAN):
    N_TEST: int = 10
    INPUT_DIM: int = 1024
    INPUT_DIM_fg: int = 1024 * 3
    OUTPUT_DIM: int = 1024 * 3
    NUM_NEURON: int = 1024 * 3
    NUM_NEURON_h: int = 1024 * 3
    TRIAL: float = 1.0
    LR_h: float = 1e-4
    LR: float = 1e-4
    N_TRAIN_SAMPLES: int = 50000
    cifar_data_path: str = "./input_data/cifar_data"
    label_result: str = "CIFAR"


@dataclass
class CfgUSPS(CfgMnist):
    save_f: int = 1
    epochs: int = 500
    INPUT_DIM: int = 64
    NUM_LAYERS_h: int = 4
    NUM_DISTRIBUTION: int = 2
    lr_schedule_per_epoch: int = 100
    N_TEST: int = 64
    OUTPUT_DIM: int = 256
    INPUT_DIM_fg: int = 256
    NUM_NEURON: int = 512
    NUM_NEURON_h: int = 512
    TRIAL: float = 1.0
    dropout: int = 0
    N_TRAIN_SAMPLES: int = 5000
    usps_mnist_path: str = "./input_data/usps_data/mnist_usps.pt"
    usps_flag: int = 1

    def get_save_path(self):
        return './data/Results_of_USPS_MNIST/distribution_{17}/h_input_dim_{5}/fg_FICNN/h_{23}/h_batchnml:{8}_dropout:{16}/convolution_{6}/layers_fg_{0}_h_{22}/neuron_fg{1}_h{21}/h_activ_{11}/lr_g{2}lr_f_{14}lr_h{15}/schedule_learning_rate:{18}/lr_schedule:{25}/gIterate_{10}_fIterate_{20}/batch_{3}/train_sample_{7}/trial_{4}_last_{13}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON,
            self.LR_g,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            'Yes' if self.convolution_flag else 'No',
            self.N_TRAIN_SAMPLES,
            'Yes' if self.batch_nml else 'No',
            self.optimizer,
            self.N_Gnet_ITERS,
            self.h_activation,
            self.positive_flag,
            self.final_actv if self.h_full_activation else 'linear',
            self.LR_f,
            self.LR_h,
            'Yes' if self.dropout else 'No',
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            'Yes' if self.hLinear else 'No',
            self.N_Fnet_ITERS,
            self.NUM_NEURON_h,
            self.NUM_LAYERS_h,
            'ResNet' if self.hResNet else 'nonResNet',
            0,
            self.lr_schedule_per_epoch
        )


@dataclass
class CfgCircSqu(Cfg3loop_F):
    opacity: float = 0.5
    scatter_size: float = 1
    square_radius: int = 4
    ring_out: int = 6
    ring_inside: int = 4
    triangle: int = 4  # this means ([(0, 4), (-4, -4), (4, -4)]
    BATCH_SIZE: int = 100
    INPUT_DIM: int = 2
    INPUT_DIM_fg: int = 2
    OUTPUT_DIM: int = 2
    NUM_NEURON: int = 10
    NUM_NEURON_h: int = 10
    NUM_LAYERS: int = 3
    NUM_LAYERS_h: int = 2
    TRIAL: float = 8.61
    NUM_DISTRIBUTION: int = 2
    LR_h: float = 1e-3
    LR_g: float = 1e-3
    LR_f: float = 1e-3
    lr_schedule: int = 20

    def get_save_path(self):
        return './data/Results_of_circ_squ/distribution_{17}/input_dim_{5}/fg_FICNN/h_{23}/h_batchnml:{8}_dropout:{16}/layers_fg_{0}_h_{22}/neuron_fg{1}_h{21}/h_activ_{11}/lr_g{2}lr_f_{14}lr_h{15}/schedule_learning_rate:{18}/lr_schedule:{25}/gIterate_{10}_fIterate_{20}/batch_{3}/train_sample_{7}/sign_{12}/trial_{4}_last_{13}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON,
            self.LR_g,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            0,
            self.N_TRAIN_SAMPLES,
            'Yes' if self.batch_nml else 'No',
            self.optimizer,
            self.N_Gnet_ITERS,
            self.h_activation,
            0,
            'hFull_activation' if self.h_full_activation else 'hFinal_linear',
            self.LR_f,
            self.LR_h,
            'Yes' if self.dropout else 'No',
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            'Yes' if self.hLinear else 'No',
            self.N_Fnet_ITERS,
            self.NUM_NEURON_h,
            self.NUM_LAYERS_h,
            'ResNet' if self.hResNet else 'nonResNet',
            0,
            self.lr_schedule_per_epoch
        )


@dataclass
class CfgBlock(Cfg3loop):
    opacity: float = 0.5
    scatter_size: float = 1
    block_side_s: int = 1
    block_side_l: int = 3
    farest_point: int = 6
    BATCH_SIZE: int = 100
    INPUT_DIM: int = 3
    INPUT_DIM_fg: int = 3
    OUTPUT_DIM: int = 3
    NUM_NEURON: int = 10
    NUM_NEURON_h: int = 10
    NUM_LAYERS: int = 3
    NUM_LAYERS_h: int = 2
    TRIAL: float = 19.1
    NUM_DISTRIBUTION: int = 2
    LR_h: float = 1e-3
    LR_g: float = 1e-3
    LR_f: float = 1e-3

    def get_save_path(self):
        return './data/Results_of_2blocks/distribution_{17}/input_dim_{5}/fg_FICNN/h_{23}/h_batchnml:{8}_dropout:{16}/layers_fg_{0}_h_{22}/neuron_fg{1}_h{21}/h_activ_{11}/lr_g{2}lr_f_{14}lr_h{15}/schedule_learning_rate:{18}/lr_schedule:{25}/gIterate_{10}_fIterate_{20}/batch_{3}/train_sample_{7}/sign_{12}/trial_{4}_last_{13}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON,
            self.LR_g,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            0,
            self.N_TRAIN_SAMPLES,
            'Yes' if self.batch_nml else 'No',
            self.optimizer,
            self.N_Gnet_ITERS,
            self.h_activation,
            0,
            'hFull_activation' if self.h_full_activation else 'hFinal_linear',
            self.LR_f,
            self.LR_h,
            'Yes' if self.dropout else 'No',
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            'Yes' if self.hLinear else 'No',
            self.N_Fnet_ITERS,
            self.NUM_NEURON_h,
            self.NUM_LAYERS_h,
            'ResNet' if self.hResNet else 'nonResNet',
            0,
            self.lr_schedule_per_epoch
        )


@dataclass
class CfgEllipse(Cfg3loop):
    BATCH_SIZE: int = 100
    INPUT_DIM: int = 1
    INPUT_DIM_fg: int = 2
    OUTPUT_DIM: int = 2
    NUM_NEURON: int = 6
    NUM_NEURON_h: int = 6
    NUM_LAYERS: int = 3
    NUM_LAYERS_h: int = 3
    TRIAL: float = 16.1
    NUM_DISTRIBUTION: int = 10
    LR_h: float = 1e-2
    LR_g: float = 1e-2
    LR_f: float = 1e-2
    N_Gnet_ITERS: int = 10
    N_Fnet_ITERS: int = 6

    def get_save_path(self):
        return './data/Results_of_ellipse/distribution_{17}/input_dim_{5}/fg_FICNN/h_{23}/h_batchnml:{8}_dropout:{16}/layers_fg_{0}_h_{22}/neuron_fg{1}_h{21}/h_activ_{11}/lr_g{2}lr_f_{14}lr_h{15}/schedule_learning_rate:{18}/lr_schedule:{25}/gIterate_{10}_fIterate_{20}/batch_{3}/train_sample_{7}/trial_{4}_last_{13}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON,
            self.LR_g,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            0,
            self.N_TRAIN_SAMPLES,
            'Yes' if self.batch_nml else 'No',
            self.optimizer,
            self.N_Gnet_ITERS,
            self.h_activation,
            0,
            'hFull_activation' if self.h_full_activation else 'hFinal_linear',
            self.LR_f,
            self.LR_h,
            'Yes' if self.dropout else 'No',
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            'Yes' if self.hLinear else 'No',
            self.N_Fnet_ITERS,
            self.NUM_NEURON_h,
            self.NUM_LAYERS_h,
            'ResNet' if self.hResNet else 'nonResNet',
            0,
            self.lr_schedule_per_epoch
        )


@dataclass
class CfgLine(Cfg3loop):
    BATCH_SIZE: int = 100
    INPUT_DIM: int = 1
    INPUT_DIM_fg: int = 2
    OUTPUT_DIM: int = 2
    NUM_NEURON: int = 6
    NUM_NEURON_h: int = 6
    NUM_LAYERS: int = 3
    NUM_LAYERS_h: int = 2
    TRIAL: float = 11
    NUM_DISTRIBUTION: int = 10
    LR_h: float = 1e-4
    LR_g: float = 1e-4
    LR_f: float = 1e-4
    N_Gnet_ITERS: int = 10
    N_Fnet_ITERS: int = 6
    hLinear: int = 1

    def get_save_path(self):
        return './data/Results_of_line/distribution_{17}/input_dim_{5}/fg_FICNN/h_{23}_Linear_{19}/h_batchnml:{8}_dropout:{16}/layers_fg_{0}_h_{22}/neuron_fg{1}_h{21}/h_activ_{11}/lr_g{2}lr_f_{14}lr_h{15}/schedule_learning_rate:{18}/lr_schedule:{25}/gIterate_{10}_fIterate_{20}/batch_{3}/train_sample_{7}/trial_{4}_last_{13}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON,
            self.LR_g,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            0,
            self.N_TRAIN_SAMPLES,
            'Yes' if self.batch_nml else 'No',
            self.optimizer,
            self.N_Gnet_ITERS,
            self.h_activation,
            0,
            'hFull_activation' if self.h_full_activation else 'hFinal_linear',
            self.LR_f,
            self.LR_h,
            'Yes' if self.dropout else 'No',
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            'Yes' if self.hLinear else 'No',
            self.N_Fnet_ITERS,
            self.NUM_NEURON_h,
            self.NUM_LAYERS_h,
            'ResNet' if self.hResNet else 'nonResNet',
            0,
            self.lr_schedule_per_epoch
        )


@dataclass
class CfgColor(Cfg3loop):
    N_TRAIN_SAMPLES: int = 2073600
    BATCH_SIZE: int = 1200
    INPUT_DIM: int = 3
    INPUT_DIM_fg: int = 3
    OUTPUT_DIM: int = 3
    NUM_NEURON: int = 16
    NUM_NEURON_h: int = 16
    NUM_LAYERS: int = 3
    NUM_LAYERS_h: int = 3
    TRIAL: float = 21
    NUM_DISTRIBUTION: int = 3
    LR_h: float = 1e-3
    LR_g: float = 1e-3
    LR_f: float = 1e-3
    color_data_path: str = "./input_data/color_transfer"

    def save_model_path(self):
        return './data/Results_of_colorTransfer/distribution_{17}/input_dim_{5}/fg_FICNN/h_{23}/h_batchnml:{8}_dropout:{16}/layers_fg_{0}_h_{22}/neuron_fg{1}_h{21}/h_activ_{11}/lr_g0.001lr_f_0.001lr_h0.001/schedule_learning_rate:{18}/lr_schedule:{25}/gIterate_{10}_fIterate_{20}/batch_{3}/train_sample_{7}/sign_{12}/trial_21_last_{13}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON,
            self.LR_g,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            0,
            self.N_TRAIN_SAMPLES,
            'Yes' if self.batch_nml else 'No',
            self.optimizer,
            self.N_Gnet_ITERS,
            self.h_activation,
            0,
            self.final_actv,
            self.LR_f,
            self.LR_h,
            'Yes' if self.dropout else 'No',
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            'Yes' if self.hLinear else 'No',
            self.N_Fnet_ITERS,
            self.NUM_NEURON_h,
            self.NUM_LAYERS_h,
            'ResNet' if self.hResNet else 'nonResNet',
            0,
            self.lr_schedule_per_epoch
        )

    def get_save_path(self):
        return './data/Results_of_colorTransfer/distribution_{17}/input_dim_{5}/fg_FICNN/h_{23}/h_batchnml:{8}_dropout:{16}/layers_fg_{0}_h_{22}/neuron_fg{1}_h{21}/h_activ_{11}/lr_g{2}lr_f_{14}lr_h{15}/schedule_learning_rate:{18}/lr_schedule:{25}/gIterate_{10}_fIterate_{20}/batch_{3}/train_sample_{7}/sign_{12}/trial_{4}_last_{13}'.format(
            self.NUM_LAYERS,
            self.NUM_NEURON,
            self.LR_g,
            self.BATCH_SIZE,
            self.TRIAL,
            self.INPUT_DIM,
            0,
            self.N_TRAIN_SAMPLES,
            'Yes' if self.batch_nml else 'No',
            self.optimizer,
            self.N_Gnet_ITERS,
            self.h_activation,
            0,
            self.final_actv,
            self.LR_f,
            self.LR_h,
            'Yes' if self.dropout else 'No',
            self.NUM_DISTRIBUTION,
            'Yes' if self.schedule_learning_rate else 'No',
            'Yes' if self.hLinear else 'No',
            self.N_Fnet_ITERS,
            self.NUM_NEURON_h,
            self.NUM_LAYERS_h,
            'ResNet' if self.hResNet else 'nonResNet',
            0,
            self.lr_schedule_per_epoch
        )


@dataclass
class CfgCuturi(Cfg3loop):
    NUM_DISTRIBUTION: int = 1
    NUM_GMM_COMPONENT: list = field(default_factory=list)
    MEAN: list = field(default_factory=list)
    COV: list = field(default_factory=list)
    INPUT_DIM: int = 2
    TRIAL: float = 20.1
    N_SAMPLES: int = 5000
    posterior_path: str = "/home/jfan97/Study_hard/barycenter/July20/barycenter_clean/input_data/bayesian_inference/bike_posterior/"


@dataclass
class CfgCuturiGMM(CfgCuturi):
    NUM_DISTRIBUTION: int = 1
    NUM_GMM_COMPONENT: list = field(default_factory=list)
    MEAN: list = field(default_factory=list)
    COV: list = field(default_factory=list)
    INPUT_DIM: int = 2
    TRIAL: float = 20.1
    N_SAMPLES: int = 1500

    def get_save_path(self):
        return './data/Results_of_Cuturi/distribution_{0}/GMM_component_{1}/input_dim_{2}/trial_{3}'.format(
            self.NUM_DISTRIBUTION,
            self.NUM_GMM_COMPONENT,
            self.INPUT_DIM,
            self.TRIAL
        )


@dataclass
class CfgCuturiBayes(CfgCuturi):
    NUM_DISTRIBUTION: int = 5
    INPUT_DIM: int = 8
    TRIAL: float = 20.1
    N_SAMPLES: int = 1500
    SCALE: int = 100
    posterior_path: str = "/home/jfan97/Study_hard/barycenter/July20/barycenter_clean/input_data/bayesian_inference/bike_posterior/"


@dataclass
class CfgCuturiMeanField(CfgCuturi):
    NUM_DISTRIBUTION: int = 5
    INPUT_DIM: int = 786
    TRIAL: float = 20.1
    N_SAMPLES: int = 1024
    SCALE: int = 1000
    test_data_path: str = 'input_data/mnist_data/mnist_17_classi_test_train.pt'
    subset_samples: int = 500

    def get_nn(self, lr_schedule=100):
        return f'data/Results_of_classification/distribution_5/digit17/input_dim_785/layers_0/neuron_{self.N_SAMPLES}/activ_celu_final_sigmoid/lr0.001/schedule_learning_rate:Yes/lr_schedule:{lr_schedule}/batch_100'


@dataclass
class CfgAverageNN(CfgCuturiMeanField):
    pass
