import glob

import numpy as np
import pybullet as p
from env_sin import *
import os
from torch import nn, optim
from shutil import copyfile
# from fast import FastNN
from fast02 import FastNN
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import torchvision


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("start", device)

class SASDATA(Dataset):
    def __init__(self, SAS_data):
        self.input_data = SAS_data[:, :28]
        self.label_data = SAS_data[:, 28:]

    def __getitem__(self, idx):

        input_data_sample = self.input_data[idx]
        label_data_sample = self.label_data[idx]
        input_data_sample = torch.from_numpy(input_data_sample).to(device, dtype=torch.float)
        label_data_sample = torch.from_numpy(label_data_sample).to(device, dtype=torch.float)
        sample = {"input": input_data_sample, "label": label_data_sample}
        return sample

    def __len__(self):
        return len(self.input_data)


def train_model(batchsize,lr):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = SASDATA(train_SAS_data)
    test_dataset = SASDATA(test_SAS_data)

    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    train_epoch_L = []
    test_epoch_L  = []
    min_loss = + np.inf

    for epoch in range(num_epoches):
        t0 = time.time()
        model.train()
        temp_l = []

        for i, bundle in enumerate(train_dataloader):
            input_d, label_d = bundle["input"],bundle["label"]

            pred_result = model.forward(input_d)
            loss = model.loss(pred_result,label_d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temp_l.append(loss.item())

        train_mean_loss = np.mean(temp_l)
        train_epoch_L.append(train_mean_loss)

        model.eval()
        temp_l = []

        with torch.no_grad():
            for i, bundle in enumerate(test_dataloader):
                input_d, label_d = bundle["input"], bundle["label"]

                pred_result = model.forward(input_d)
                loss = model.loss(pred_result, label_d)
                temp_l.append(loss.item())

            test_mean_loss = np.mean(temp_l)
            test_epoch_L.append(test_mean_loss)

        if test_mean_loss < min_loss:
            # print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(train_mean_loss))
            # print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(test_mean_loss))
            min_loss = test_mean_loss
            PATH = log_path + '/best_model.pt'
            torch.save(model.state_dict(), PATH)
        else:
            print("May Overfitting...")
        np.savetxt(log_path + "training_L.csv", np.asarray(train_epoch_L))
        np.savetxt(log_path + "testing_L.csv", np.asarray(test_epoch_L))

        t1 = time.time()
        # print(epoch, "time used: ", (t1 - t0) / (epoch + 1), "training mean loss: ",train_mean_loss, "lr:", lr)
        print(epoch, "training loss: ",train_mean_loss, "Test loss: ", test_mean_loss)


def valid_model(batchsize):
    test_dataset = SASDATA(test_SAS_data)

    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    n = 0
    with torch.no_grad():
        for i, bundle in enumerate(test_dataloader):
            n += 1
            input_d, label_d = bundle["input"], bundle["label"]

            pred_result = model.forward(input_d)
            if i == 0:
                all_loss = abs(pred_result- label_d)[0]
            else:
                all_loss += abs(pred_result- label_d)[0]

    all_loss = all_loss / n
    print(all_loss, n)


def compare_model_and_sim(env, para_path, sm_model, num_steps=50, norm_scale=0.1):
    obs = env.reset()
    predict = []
    sim = []
    para_data = np.loadtxt(para_path)
    for i_steps in range(num_steps):
        # sin_para = random_para()

        # sas = random.choice(sas_data)
        # para = sas[18:28]
        para = np.random.normal(para_data, norm_scale)
        SA = np.hstack(([obs],[para]))
        SA = torch.from_numpy(SA.astype(np.float32)).to(device)
        obs, r, done, _ = env.step(para)
        pred_ns = sm_model.forward(SA)
        pred_ns_numpy = pred_ns.cpu().detach().numpy()
        predict.append(pred_ns_numpy[0])
        sim.append(obs)
    
    np.savetxt(log_path+'predict.csv', np.asarray(predict))
    np.savetxt(log_path+'sim.csv', np.asarray(sim))



            


def test_model(env, sm_model, epoch_num = 5,TASK = 'f'):

    num_choices = 10000

    for epoch in range(epoch_num):
        print("epoch:", epoch)
        fail = 0
        result = 0

        obs = env.reset()
        for step_i in range(10):
            cur_state = [obs]*num_choices


            action_input = []
            for i in range(num_choices):
                action_input.append(random_para())
            action_input = np.asarray(action_input)
            cur_state = np.asarray(cur_state)
            SA = np.hstack((cur_state,action_input))

            SA = torch.from_numpy(SA.astype(np.float32)).to(device)

            pred_ns = sm_model.forward(SA)
            pred_ns_numpy = pred_ns.cpu().detach().numpy()

            if TASK == "f":
                all_a_rewards =  2 * pred_ns_numpy[:, 1] - abs(pred_ns_numpy[:, 0]) - abs(pred_ns_numpy[:,3])- abs(pred_ns_numpy[:,4])- abs(pred_ns_numpy[:,5])
            elif TASK == "l":
                all_a_rewards = pred_ns_numpy[:, 5]  # - abs(pred_ns_numpy[:,1])
            elif TASK == "r":
                all_a_rewards = -pred_ns_numpy[:, 5]  # - abs(pred_ns_numpy[:,1])
            elif TASK == "stop":
                all_a_rewards = 10 - abs(pred_ns_numpy[:, 1]) - abs(pred_ns_numpy[:, 0])
            elif TASK == "move_r":
                all_a_rewards = pred_ns_numpy[:, 0]
            elif TASK == "b":
                all_a_rewards = - 2 * pred_ns_numpy[:, 1] - abs(pred_ns_numpy[:, 0])
            else:
                all_a_rewards = np.zeros(num_choices)

            greedy_select = int(np.argmax(all_a_rewards))

            choose_a = action_input[greedy_select]
            obs, r, done, _ = env.step(choose_a)

            print(env.robot_location())
            result += r


if __name__ == '__main__':

    model = FastNN(18, 10).to(device)
    # log_path = "software/sm_516(20k)/"

    EXAMPLE_PATH_LIST = glob.glob('dataset/robot_dataset/example/*/', recursive = True)
    print(EXAMPLE_PATH_LIST)
    robot_example_id = 0
    robot_name = EXAMPLE_PATH_LIST[robot_example_id].split('/')[-2]
    print(robot_name)


    # ROBOTID = 516
    # robot_name = standRobot('dataset/RoboId(good_gait).txt', ROBOTID)

    MODE = 0
    Pretrain = False

    # Train self-model
    if MODE == 0:
        num_data = 100000
        data_file_name = 100000  # may different with num data
        model_name = "norm-02-onepara"
        # model_name = "rand"
        log_path = "dataset/10-robots-data/" + robot_name + "/model/%s_%d-log/"%(model_name,num_data)
        try:
            os.mkdir(log_path)
        except:
            pass
        all_data = np.loadtxt("dataset/10-robots-data/" + robot_name + '/data/%s/%d_sas-10.csv'%(model_name,data_file_name))
        print(all_data.shape)
        training_data_num = int(num_data*0.8)
        train_SAS_data = all_data[:training_data_num]
        test_SAS_data = all_data[training_data_num:num_data]


        lr = 1e-4
        batchsize =   64
        num_epoches = 10000

        try:
            os.mkdir(log_path)
        except OSError:
            pass

        if Pretrain:
            # pretrain_model_pth = "software/sm_516(2)/best_model.pt"
            pretrain_model_pth = "dataset/select_para/sm_516(2)/best_model.pt"
            model.load_state_dict(torch.load(pretrain_model_pth))
            model.to(device)

        train_model(batchsize,lr)


    elif MODE == 1:
        model.load_state_dict(torch.load(log_path + "best_model.pt"))
        model.to(device)

        physicsClient = p.connect(p.GUI)

        robot_info = getRobotURDFAndInitialJointAngleFilePath(
            robot_name, urdf_path=URDF_JOINT_FILE_PATH)

        env = RobotEnv(robot_info, [0, 0, 0.3], follow_robot=False)

        test_model(env, sm_model = model)

    elif MODE == 2:
        test_SAS_data = np.loadtxt('dataset/select_para/compare_rand_gaus/%s/norm-01/40000_sas-10.csv' % robot_name)
        model.load_state_dict(torch.load(log_path + "best_model.pt"))
        model.to(device)
        batchsize =   1
        valid_model(batchsize)  

    elif MODE == 3:
        model.load_state_dict(torch.load(log_path + "best_model.pt"))
        model.to(device)

        physicsClient = p.connect(p.GUI)

        robot_info = getRobotURDFAndInitialJointAngleFilePath(
            robot_name, urdf_path=URDF_JOINT_FILE_PATH)

        env = RobotEnv(robot_info, [0, 0, 0.3], follow_robot=False)
        para_path = 'dataset/select_para/compare_rand_gaus/%s/norm-01/40000_sas-10.csv' % robot_name
        compare_model_and_sim(env=env, para_path=para_path, sm_model = model)




