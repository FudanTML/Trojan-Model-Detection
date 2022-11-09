# Trojan-Model-Detection

### Problem Statement

Backdoor attacks have emerged as the main threat to the safe deployment of DNNs. An adversary poisons some of the training data in such attacks by installing a trigger. The goal is to make the trained DNN output the attacker's desired class whenever the trigger is activated while performing as usual for clean data. Various approaches have recently been proposed to detect/defend malicious backdoored DNNs. However, how to efficiently defend against this attack is still an open question. In this project, we would like you to focus on the project related to backdoor defense:

### Project Goals  

- **Trojan Model Detection:** Given an untrusted pre-trained network, the goal of trojan detection is to reveal the potential backdoored model (and to identify the infected label, i.e. backdoor related label). Neural cleanse (NC) [1] has proposed a novel and generalizable technique for detecting and reverse engineering hidden triggers embedded inside deep neural networks. Early work suggests that standard Trojan attacks may be easy to detect, but recently it has been shown that in simple cases one can design practically undetectable Trojans [2, 3]. So in this project, we invite you to help answer an important research question of how hard is it to detect hidden functionality that is trying to stay hidden? And we encourage you to design or explore a new trojan detection technique against the advanced backdoor attacks.


### Trojan Model Zoo

We have randomly sampled a subset of the backdoored model from the Trojan Detection Challenge [4]. You can download from here [link].

### Example of NC

For a quick start, we have provided an example of trojan model detection by NC in folder. 

```
def reverse_engineer(model_id, para_lamda):
    param = {
        "dataset": "cifar10",
        "Epochs": 20,
        "batch_size": 64,
        "lamda": para_lamda,
        "num_classes": 10,
        "image_size": (32, 32),
        "ratio": 0.005
    }
    model = torch.load("/media/server/8961e245-931a-4871-9f74-9df58b1bd938/server/lyg/tdc-starter-kit-main/tdc_datasets/target_label_prediction/train/id-"+str(model_id)+"/model.pt").to(device)

    
    tf_train = torchvision.transforms.Compose([  
        torchvision.transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='/media/server/8961e245-931a-4871-9f74-9df58b1bd938/server/lyg/CIFAR10_GTSRB_dynamic_and_flooding/flooding/data', train=False, download=False)

    list_ = []
    count = [0]*param["num_classes"]
    threshold = int(param["ratio"]*len(train_dataset))
    for i in train_dataset:
        if count[i[1]] < threshold:
            list_.append(i)
            count[i[1]] += 1
    train_dataset = list_
    count = [0]*param["num_classes"]

    train_dataset = augDataset_npy(full_dataset=train_dataset, transform=tf_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size = param["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
)
    

    for label in range(param["num_classes"]):
        trigger, mask = train(model, label, train_loader, param)
        norm_list.append(mask)

        trigger = trigger.cpu().detach().numpy()
        trigger = np.transpose(trigger, (1,2,0))
        plt.axis("off")
        plt.imshow(trigger)
        plt.savefig('mask/trigger_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)

        mask = mask.cpu().detach().numpy()
        plt.axis("off")
    
    l1_norm_list = norm_list[-param["num_classes"]:]
    list_target = [0]*param["num_classes"]
    for i in range(len(list_target)):
        list_target[i] = i
    flag_label = outlier_detection(l1_norm_list, list_target)
        
    
    print("flag label:"+str(flag_label))
    return flag_label
```


### Reference 

[1] Wang B, Yao Y, Shan S, et al. Neural cleanse: Identifying and mitigating backdoor attacks in neural networks[C]//2019 IEEE Symposium on Security and Privacy (SP). IEEE, 2019: 707-723.  
[2] Nguyen A, Tran A. WaNet--Imperceptible Warping-based Backdoor Attack[J]. ICLR, 2021.  
[3] Nguyen T A, Tran A. Input-aware dynamic backdoor attack[J]. NeurIPS, 2021.  
[4] Trojan Detection Challenge, https://trojandetection.ai/.
