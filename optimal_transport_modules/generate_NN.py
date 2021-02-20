from __future__ import print_function
import torch
import torch.utils.data
import torch.nn as nn
import optimal_transport_modules.icnn_modules as NN_modules


# * fixed weight -> ICNN

def generate_FixedWeight_NN(cfg):
    convex_f, convex_g = generate_FixedWeight_fg_NN(cfg)
    generator_h = generate_FixedWeight_h_NN(cfg)
    return convex_f, convex_g, generator_h


def generate_FixedWeight_fg_NN(cfg):
    convex_f = generate_icnn_fg_NN(cfg, choice='f')
    convex_g = generate_icnn_fg_NN(cfg, choice='g')
    return convex_f, convex_g


def generate_icnn_fg_NN(cfg, choice):
    if cfg.INPUT_DIM_fg <= 0:
        cfg.INPUT_DIM_fg = cfg.INPUT_DIM
    nn_list = nn.ModuleList(
        [NN_modules.ICNN_LastInp_Quadratic(
            cfg.INPUT_DIM_fg,
            cfg.NUM_NEURON,
            cfg.fg_activation,
            cfg.NUM_LAYERS) for i in range(
            cfg.NUM_DISTRIBUTION)])

    if cfg.load_fg == True:
        model_load_path = cfg.save_model_path()
        for idx in range(cfg.NUM_DISTRIBUTION):
            nn_list[idx] = load_generator_fg(
                model_load_path, idx + 1, nn_list[idx], cfg.epochs, choice)
        return nn_list
    else:
        return nn_list


def generate_FixedWeight_h_NN(args):
    if args.hLinear:
        return NN_modules.Average_Weights_Linear(
            args.INPUT_DIM, args.OUTPUT_DIM, args.NUM_NEURON_h, args.NUM_LAYERS_h)
    else:
        generator_h = NN_modules.Average_Weights_NormalNet(
            args.INPUT_DIM, args.OUTPUT_DIM, args.NUM_NEURON_h, args.h_activation, args.NUM_LAYERS_h, args.batch_nml, args.dropout, final_actv=args.final_actv)
        if args.load_h == False:
            return generator_h
        else:
            model_load_path = args.save_model_path()
            # model_load_path = f'/home/jfan97/Study_hard/barycenter/July20/barycenter_clean/data/Results_of_colorTransfer/distribution_3/input_dim_3/fg_FICNN/h_nonResNet/h_batchnml:Yes_dropout:No/layers_fg_{args.NUM_LAYERS}_h_{args.NUM_LAYERS_h}/neuron_fg{args.NUM_NEURON}_h{args.NUM_NEURON_h}/h_activ_Prelu/lr_g0.001lr_f_0.001lr_h{args.LR_h}/schedule_learning_rate:Yes/lr_schedule:20/gIterate_6_fIterate_4/batch_1200/train_sample_2073600/sign_0/trial_21.2_last_{args.final_actv}'
            generator_h = load_generator_h(
                model_load_path, generator_h, args.epochs)
            return generator_h

# * unfixed weight -> PICNN


def generate_expanded_picnn_fg_NN(cfg):
    return nn.ModuleList(
        [NN_modules.PICNN_expanded(
            cfg.INPUT_DIM, cfg.NUM_DISTRIBUTION,
            cfg.NUM_NEURON_fg_weight,
            cfg.NUM_NEURON_fg_sample,
            cfg.fg_activation,
            cfg.NUM_LAYERS_fg) for i in range(
            cfg.NUM_DISTRIBUTION)])


def generate_normal_picnn_fg_NN(cfg):
    return nn.ModuleList(
        [NN_modules.PICNN_LastInp_Quadratic(
            cfg.INPUT_DIM, cfg.NUM_DISTRIBUTION,
            cfg.NUM_NEURON_fg_weight,
            cfg.NUM_NEURON_fg_sample,
            cfg.fg_activation,
            cfg.NUM_LAYERS_fg) for i in range(
            cfg.NUM_DISTRIBUTION)])


def generate_UnfixedWeight_fg_NN(cfg):
    if cfg.expanded_PICNN:
        convex_f = generate_expanded_picnn_fg_NN(cfg)
        convex_g = generate_expanded_picnn_fg_NN(cfg)
    else:
        convex_f = generate_normal_picnn_fg_NN(cfg)
        convex_g = generate_normal_picnn_fg_NN(cfg)
    return convex_f, convex_g


def generate_UnfixedWeight_h_NN(cfg):
    if cfg.h_PICNN_flag:
        generator_h = NN_modules.Different_Weights_PICNN(
            cfg.INPUT_DIM, cfg.NUM_DISTRIBUTION, cfg.NUM_NEURON_h_weight, cfg.NUM_NEURON_h_sample, cfg.h_activation, cfg.NUM_LAYERS_h, cfg.h_full_activation)
    # elif cfg.TRIAL == 8.5:
    #     generator_h = NN_modules.Different_Weights_Yongxin(
    #         cfg.INPUT_DIM, cfg.OUTPUT_DIM, cfg.NUM_DISTRIBUTION, cfg.NUM_NEURON_h, cfg.h_activation, cfg.NUM_LAYERS_h, cfg.batch_nml, cfg.dropout, cfg.h_full_activation)
    else:
        generator_h = NN_modules.Different_Weights_NormalNet(
            cfg.INPUT_DIM, cfg.OUTPUT_DIM, cfg.NUM_DISTRIBUTION, cfg.NUM_NEURON_h, cfg.h_activation, cfg.NUM_LAYERS_h, cfg.batch_nml, cfg.dropout, cfg.h_full_activation)
    return generator_h


def generate_UnfixedWeight_NN(cfg):
    convex_f, convex_g = generate_UnfixedWeight_fg_NN(cfg)
    generator_h = generate_UnfixedWeight_h_NN(cfg)
    return convex_f, convex_g, generator_h

# * fully connected


def generate_fully_connected(cfg):
    return NN_modules.Fully_connected(cfg.INPUT_DIM, cfg.OUTPUT_DIM, cfg.NUM_NEURON, cfg.NUM_LAYERS, cfg.activation, cfg.final_actv)

#! load


def load_generator_h(results_save_path, generator_h, epochs, device=None):
    model_save_path = results_save_path + '/storing_models'
    generator_h.load_state_dict(torch.load(
        model_save_path + '/generator_h_epoch{0}.pt'.format(epochs), map_location=device))

    # try:
    #     generator_h.load_state_dict(torch.load(
    #         model_save_path + '/generator_h_epoch{0}.pt'.format(epochs), map_location=device))
    # except:
    #     print("no such file")
    #     print(model_save_path + '/generator_h_epoch{0}.pt'.format(epochs))

    return generator_h.cuda(device)


def load_generator_fg(results_save_path, idx, generator_g, epochs, choice='g', device=None):
    model_save_path = results_save_path + '/storing_models'
    try:
        generator_g.load_state_dict(torch.load(
            model_save_path + f'/{choice}{idx}_epoch{epochs}.pt', map_location=device))
    except:
        print("no file for f/g network")
        print(model_save_path)
    return generator_g.cuda(device)


def load_fully_connected(results_save_path, neural_network, epochs, device):
    model_save_path = results_save_path + '/storing_models'
    neural_network.load_state_dict(torch.load(
        model_save_path + '/nn_2layer_epoch{0}.pt'.format(epochs)))
    return neural_network.cuda(device)


def load_2layer_from_barycenter(neural_network, loaded_tensor, device):
    neural_network.layer1.weight = torch.nn.Parameter(loaded_tensor[:, :-1])
    neural_network.last_layer.weight = torch.nn.Parameter(
        loaded_tensor[:, -1].reshape(1, -1))
    return neural_network.cuda(device)
